#include <fstream>
#include <stdexcept>
#include "json.hpp"
#include "Model.h"
#include "Server.h"
#include "SchedulingPolicy.h"
#include "Request.h"
#include <iostream>

using namespace std;
using json = nlohmann::json;

Config Server::readConfig(string configPath) const
{
    std::ifstream input(configPath);
    json config;
    input >> config;
    // All the GPUs have the same block size.
    size_t blockSize = config["block_size"].get<size_t>();
    vector<GPUConfig> gpuConfigs;
    for (auto &gpu : config["gpus"])
    {
        size_t memoryBytes = gpu["memory_bytes"].get<size_t>();
        GPUConfig gpuConfig{memoryBytes};
        gpuConfigs.push_back(gpuConfig);
    }
    vector<ModelConfig> modelConfigs;
    for (auto &model : config["models"])
    {
        string id = model["id"].get<string>();
        size_t embeddingSize = model["embedding_size"].get<size_t>();
        ModelConfig modelConfig{id, embeddingSize};
        modelConfigs.push_back(modelConfig);
    }
    vector<RequestConfig> requestConfigs;
    for (auto &request : config["requests"])
    {
        size_t id = request["id"].get<size_t>();
        string typeStr = request["type"].get<string>();
        if (typeStr != "generative" && typeStr != "text_embedder")
        {
            throw invalid_argument("invalid request type");
        }
        RequestType type = typeStr == "generative" ? RequestType::Generative : RequestType::TextEmbedder;
        string modelId = request["model"].get<string>();
        int admissionCycle = request["cycle"].get<int>();
        string prompt = request["prompt"].get<string>();
        string scriptedOutput = "";
        if (request.count("scripted_output") > 0)
        {
            scriptedOutput = request["scripted_output"].get<string>();
        }
        size_t maxOutputTokens = 0;
        if (request.count("max_output_tokens") > 0)
        {
            maxOutputTokens = request["max_output_tokens"].get<size_t>();
        }
        RequestConfig reqConfig{id, type, modelId, admissionCycle, prompt, scriptedOutput, maxOutputTokens};
        requestConfigs.push_back(reqConfig);
    }

    if (config.count("scheduling_policy") == 0)
    {
        throw invalid_argument("missing scheduling policy");
    }
    string policy = config["scheduling_policy"].get<string>();

    return Config{blockSize, gpuConfigs, modelConfigs, requestConfigs, policy};
}

void Server::serve()
{
    while (step())
    {
        currentCycle++;
    }
    //TODO: erase before submission
    std::cout << "Active requests: " << activeRequests.size()
              << "Waiting requests: " << waitingRequests.size()
              << "Non-submitted requests: " << nonSubmittedRequests.size()
              << "Cycles: " << currentCycle
              << std::endl;
    
    int cacheHits=0;
    int evictions=0;
    for (int gpuIndex = 0; gpuIndex < gpus.size(); gpuIndex++)
    {
        cacheHits +=gpus[gpuIndex].getCacheHits();
        evictions +=gpus[gpuIndex].getEvictions(); 
    }
    std::cout<<"summary: cycles="<< currentCycle
    <<" cache_hits="<< cacheHits
    <<" evictions="<<evictions
    << std::endl;
}
bool Server::step()
{
    if (nonSubmittedRequests.empty() && waitingRequests.empty() && activeRequests.empty())
    {
        return false;
    }

    for (auto request = nonSubmittedRequests.begin(); request != nonSubmittedRequests.end();)
    {
        if ((*request)->getAdmissionCycle() == currentCycle)
        {
            waitingRequests.push_back(*request);
            request = nonSubmittedRequests.erase(request);
        }
        else
        {
            ++request;
        }
    }

    std::vector<const Request *> eligible(waitingRequests.begin(), waitingRequests.end());
    std::vector<int> order = policy->choose(eligible, gpus);
    for (auto requestIndex : order)
    {

        int theGPUIndex = -1;
        auto theprefix = PrefixMatch{0, std::vector<std::size_t>()};
        for (int gpuIndex = 0; gpuIndex < gpus.size(); gpuIndex++)
        {
            auto gpu = &gpus[gpuIndex];
            if (gpu->getBlocksForMemory(waitingRequests[requestIndex]->getNeededMemory(*gpu)) > gpu->availableSlots())
            {
                continue;
            }
            if (theGPUIndex != -1)
            {
                theGPUIndex = gpuIndex;
                theprefix = gpu->findLongestCachedPrefix(waitingRequests[requestIndex]->getModel().getId(),
                                                         waitingRequests[requestIndex]->getModel().getEmbeddingSize(), waitingRequests[requestIndex]->getCacheableTokens());

                continue;
            }
            auto prefix = gpu->findLongestCachedPrefix(waitingRequests[requestIndex]->getModel().getId(),
                                                       waitingRequests[requestIndex]->getModel().getEmbeddingSize(), waitingRequests[requestIndex]->getCacheableTokens());
            if (prefix.matchedTokens > theprefix.matchedTokens)
            {
                theprefix = prefix;
                theGPUIndex = gpuIndex;
            }
            else if (prefix.matchedTokens == theprefix.matchedTokens)
            {
                if (gpu->availableSlots() > gpus[theGPUIndex].availableSlots())
                {
                    theprefix = prefix;
                    theGPUIndex = gpuIndex;
                }
                else if (gpu->availableSlots() == gpus[theGPUIndex].availableSlots())
                {
                    if (gpuIndex < theGPUIndex)
                    {
                        theprefix = prefix;
                        theGPUIndex = gpuIndex;
                    }
                }
            }
            if (theGPUIndex == -1)
            {
                std:cout<< "rejected " <<waitingRequests[requestIndex]->getId()
                <<" cycle="<< currentCycle << std::endl;
                delete waitingRequests[requestIndex];
                waitingRequests.erase(waitingRequests.begin() + requestIndex);
                
            }
            else
            {
                waitingRequests[requestIndex]->setChoosenGPU(theGPUIndex);
                for (gpuIndex = 0; gpuIndex < gpus.size(); gpuIndex++)
                {
                    if (gpu[gpuIndex].findLongestCachedPrefix(waitingRequests[requestIndex]->getModel().getId(),
                                                         waitingRequests[requestIndex]->getModel().getEmbeddingSize(), waitingRequests[requestIndex]->getCacheableTokens()).matchedTokens > theprefix.matchedTokens)
                    {   
                        theprefix = gpu[gpuIndex].findLongestCachedPrefix(waitingRequests[requestIndex]->getModel().getId(),
                                                                          waitingRequests[requestIndex]->getModel().getEmbeddingSize(), waitingRequests[requestIndex]->getCacheableTokens());
                        gpus[theGPUIndex].copyMemoryFromOtherGPU(gpus[gpuIndex], theprefix.blockIds);
                    }
                }
                waitingRequests[requestIndex]->reserveMemory(gpus[theGPUIndex]);
                activeRequests.push_back(waitingRequests[requestIndex]);
                waitingRequests.erase(waitingRequests.begin() + requestIndex);
            }
            
        }
    }
    for (auto request = activeRequests.begin(); request != activeRequests.end();)
    {
        (*request)->act(gpus[(*request)->getChoosenGPU()]);
        if ((*request)->finished())
        {
            (*request)->deallocateMemory(gpus[(*request)->getChoosenGPU()]);
            std:cout<< "finished " <<(*request)->getId()
                <<" cycle="<< currentCycle << 
                " gpu=" <<(*request)->getChoosenGPU() <<
                ": "<< (*request)->getOutput()<< std::endl;
                delete *request;
            request = activeRequests.erase(request);
        }
        else
            ++request;
    }

    return true;
}
Server::Server(string configPath)
    : gpus(), models(), nonSubmittedRequests(), waitingRequests(), activeRequests(), policy(nullptr), currentCycle(0)
{
    Config config = readConfig(configPath);
    gpus.reserve(config.gpus.size());
    for (const auto &gpuConfig : config.gpus)
    {
        gpus.emplace_back(gpuConfig.memoryBytes, config.blockSize);
    }
    models.reserve(config.models.size());
    for (const auto &modelConfig : config.models)
    {
        models.emplace_back(modelConfig.id, modelConfig.embeddingSize);
    }
    
    if (config.policy == "arrivalOrder")
    {
        policy = new ArrivalOrderPolicy();
    }
    else if (config.policy == "shortestJobFirst")
    {
        policy = new ShortestJobPolicy();
    }
    else if (config.policy == "longestPrefixFirst")
    {
        policy = new LongestPrefixPolicy();
    }
    else
    {
        throw invalid_argument("invalid scheduling policy");
    }

    nonSubmittedRequests.reserve(config.requests.size());
    for (const auto &requestConfig : config.requests)
    {
        const Model *model = nullptr;
        for (const auto &m : models)
        {
            if (m->getId() == requestConfig.modelId)
            {
                model = m;
                break;
            }
        }
        if (requestConfig.type == RequestType::Generative)
        {
            nonSubmittedRequests.push_back(new GenerativeRequest(requestConfig.id, *model, requestConfig.admissionCycle, requestConfig.prompt, requestConfig.scriptedOutput, requestConfig.maxOutputTokens));
        }
        else if (requestConfig.type == RequestType::TextEmbedder)
        {
            nonSubmittedRequests.push_back(new EmbeddingRequest(requestConfig.id, *model, requestConfig.admissionCycle, requestConfig.prompt));
        }
    }
}

Server::~Server()
{
    clear();
}
void Server::clear()
{
    for (auto request : nonSubmittedRequests)
    {
        delete request;
    }
    for (auto request : waitingRequests)
    {
        delete request;
    }
    for (auto request : activeRequests)
    {
        delete request;
    }
    delete policy;
    for(auto model:)
}
Server::Server(const Server &other)
    : gpus(other.gpus), models(other.models), nonSubmittedRequests(),
      waitingRequests(), activeRequests(), policy(other.policy->clone())
{

    for (const auto &request : other.nonSubmittedRequests)
    {
        nonSubmittedRequests.push_back(request->clone(request->getModel()));
    }
    for (const auto &request : other.waitingRequests)
    {
        waitingRequests.push_back(request->clone(request->getModel()));
    }
    for (const auto &request : other.activeRequests)
    {
        activeRequests.push_back(request->clone(request->getModel()));
    }
}

Server &Server::operator=(const Server &other)
{
    if (this != &other)
    {
        clear();
        gpus = other.gpus;
        models = other.models;
        policy = other.policy->clone();

        for (const auto &request : other.nonSubmittedRequests)
        {
            nonSubmittedRequests.push_back(request->clone(request->getModel()));
        }
        for (const auto &request : other.waitingRequests)
        {
            waitingRequests.push_back(request->clone(request->getModel()));
        }
        for (const auto &request : other.activeRequests)
        {
            activeRequests.push_back(request->clone(request->getModel()));
        }
    }
    return *this;
}

Server::Server(Server &&other) noexcept
    : gpus(std::move(other.gpus)), models(std::move(other.models)),
      nonSubmittedRequests(std::move(other.nonSubmittedRequests)),
      waitingRequests(std::move(other.waitingRequests)), activeRequests(std::move(other.activeRequests)),
      policy(other.policy)
{
    other.policy = nullptr;
}

Server &Server::operator=(Server &&other) noexcept
{
    if (this != &other)
    {
        clear();
        gpus = std::move(other.gpus);
        models = std::move(other.models);
        nonSubmittedRequests = std::move(other.nonSubmittedRequests);
        waitingRequests = std::move(other.waitingRequests);
        activeRequests = std::move(other.activeRequests);
        policy = other.policy;
        other.policy = nullptr;
    }
    return *this;
}