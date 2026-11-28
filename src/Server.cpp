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
    // TODO: erase before submission
    // std::cout << "Active requests: " << activeRequests.size()
    //           << "Waiting requests: " << waitingRequests.size()
    //           << "Non-submitted requests: " << nonSubmittedRequests.size()
    //           << "Cycles: " << currentCycle
    //           << std::endl;

    int cacheHits = 0;
    int evictions = 0;
    for (std::size_t gpuIndex = 0; gpuIndex < gpus.size(); gpuIndex++)
    {
        cacheHits += gpus[gpuIndex].getCacheHits();
        evictions += gpus[gpuIndex].getEvictions();
    }
    std::cout << "summary: cycles=" << currentCycle
              << " cache_hits=" << cacheHits
              << " evictions=" << evictions
              << std::endl;
}
bool Server::step()
{
    if (nonSubmittedRequests.empty() && waitingRequests.empty() && activeRequests.empty())
    {
        return false;
    }

    //    take the request to waiting
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

    // use the policy
    std::vector<const Request *> eligible(waitingRequests.begin(), waitingRequests.end());
    std::vector<int> order = policy->choose(eligible, gpus);

    std::vector<bool> handled(waitingRequests.size(), false);

    // for each waiting reauest
    for (auto requestIndex : order)
    {
        handled[requestIndex] = true;
        Request &currReq = *waitingRequests[requestIndex];
        int chosenGPUIndex = -1;
        auto maxLengthPrefix = PrefixMatch{0, std::vector<std::size_t>()};
        std::size_t neededBlocksInChosenGPU = 0;

        // finds the best gpu
        for (std::size_t gpuIndex = 0; gpuIndex < gpus.size(); gpuIndex++)
        {
            const GPU &gpu = gpus[gpuIndex];

            auto prefix = gpu.findLongestCachedPrefix(currReq.getModel().getId(),
                                                      currReq.getModel().getEmbeddingSize(),
                                                      currReq.getCacheableTokens());
            // finds if gpu has enought slots
            std::size_t neededBlocks = gpu.getBlocksForMemory(currReq.getNeededMemory(gpu)) + gpu.countIdles(prefix);
            if (neededBlocks > gpu.availableSlots())
            {
                continue;
            }

            // if this the first gpu that fits the request take him and continue
            if (chosenGPUIndex == -1)
            {
                neededBlocksInChosenGPU = neededBlocks;
                chosenGPUIndex = gpuIndex;
                maxLengthPrefix = prefix;

                continue;
            }

            // check if the current GPU's longest prefix is longer
            if (prefix.matchedTokens > maxLengthPrefix.matchedTokens)
            {
                maxLengthPrefix = prefix;
                chosenGPUIndex = gpuIndex;
                neededBlocksInChosenGPU = neededBlocks;
            }
            else if (prefix.matchedTokens == maxLengthPrefix.matchedTokens)
            {
                // check if in new GPU need fewer new blocks
                if (neededBlocks < neededBlocksInChosenGPU)
                {
                    neededBlocksInChosenGPU = neededBlocks;
                    maxLengthPrefix = prefix;
                    chosenGPUIndex = gpuIndex;
                }
            }
        }

        // if request can't be processed by any GPU, reject it
        if (chosenGPUIndex == -1)
        {
            std::cout << "rejected " << currReq.getId()
                      << " cycle=" << currentCycle << std::endl;
            delete waitingRequests[requestIndex];
            waitingRequests[requestIndex] = nullptr;
        }
        else
        {
            // if there is gpu with a better prefix for the requst copy this prefix
            currReq.setChoosenGPU(chosenGPUIndex);
            for (std::size_t gpuIndex = 0; gpuIndex < gpus.size(); gpuIndex++)

            {
                if (gpus[gpuIndex].findLongestCachedPrefix(currReq.getModel().getId(),
                                                           currReq.getModel().getEmbeddingSize(), currReq.getCacheableTokens())
                        .matchedTokens > maxLengthPrefix.matchedTokens)
                {
                    maxLengthPrefix = gpus[gpuIndex].findLongestCachedPrefix(currReq.getModel().getId(),
                                                                             currReq.getModel().getEmbeddingSize(),
                                                                             currReq.getCacheableTokens());
                    gpus[chosenGPUIndex].copyMemoryFromOtherGPU(gpus[gpuIndex], maxLengthPrefix.blockIds);
                }
            }
            // move the request to active requests
            currReq.reserveMemory(gpus[chosenGPUIndex]);
            activeRequests.push_back(waitingRequests[requestIndex]);
            waitingRequests[requestIndex] = nullptr;
        }
    }
    std::vector<Request *> nextWaiting;
    for (std::size_t i = 0; i < waitingRequests.size(); ++i)
    {
        if (!handled[i])
            nextWaiting.push_back(waitingRequests[i]);
    }
    waitingRequests = std::move(nextWaiting);
    // runs the active request
    for (auto request = activeRequests.begin(); request != activeRequests.end();)
    {
        (*request)->act(gpus[(*request)->getChoosenGPU()]);
        if ((*request)->finished())
        {
            (*request)->deallocateMemory(gpus[(*request)->getChoosenGPU()]);
            std::string output = ((*request)->getOutput() == "") ? "" : " " + (*request)->getOutput();
            std::cout << "finished " << (*request)->getId()
                      << " cycle=" << currentCycle << " gpu=" << (*request)->getChoosenGPU() << ":" << output << std::endl;
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
        models.push_back(new Model(modelConfig.id, modelConfig.embeddingSize));
    }

    if (config.policy == "arrivalOrder")
    {
        policy = new ArrivalOrderPolicy();
    }
    else if (config.policy == "shortestJob")
    {
        policy = new ShortestJobPolicy();
    }
    else if (config.policy == "longestPrefix")
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
    nonSubmittedRequests.clear();

    for (auto request : waitingRequests)
    {
        delete request;
    }
    waitingRequests.clear();

    for (auto request : activeRequests)
    {
        delete request;
    }
    activeRequests.clear();

    delete policy;
    policy = nullptr;
    for (auto model : models)
    {
        delete model;
    }
    models.clear();
}
Server::Server(const Server &other)
    : gpus(other.gpus), models(), nonSubmittedRequests(),
      waitingRequests(), activeRequests(), policy(other.policy->clone()), currentCycle(other.currentCycle)
{
models.reserve(other.models.size());
for (const Model *m : other.models)
{
    models.push_back(new Model(m->getId(), m->getEmbeddingSize()));
}

for (const auto &request : other.nonSubmittedRequests)
{
    const Model *targetModel = nullptr;
    for (const Model *m : models)
    {
        if (m->getId() == request->getModel().getId())
        {
            targetModel = m;
            break;
        }
    }
    nonSubmittedRequests.push_back(request->clone(*targetModel));
}

for (const auto &request : other.waitingRequests)
{
    const Model *targetModel = nullptr;
    for (const Model *m : models)
    {
        if (m->getId() == request->getModel().getId())
        {
            targetModel = m;
            break;
        }
    }
    waitingRequests.push_back(request->clone(*targetModel));
}

for (const auto &request : other.activeRequests)
{
    const Model *targetModel = nullptr;
    for (const Model *m : models)
    {
        if (m->getId() == request->getModel().getId())
        {
            targetModel = m;
            break;
        }
    }
    activeRequests.push_back(request->clone(*targetModel));
}
}

Server &Server::operator=(const Server &other)
{
    if (this != &other)
    {
        clear();
        gpus = other.gpus;
        policy = other.policy->clone();
        currentCycle = other.currentCycle;
        models.reserve(other.models.size());
for (const Model *m : other.models)
{
    models.push_back(new Model(m->getId(), m->getEmbeddingSize()));
}

for (const auto &request : other.nonSubmittedRequests)
{
    const Model *targetModel = nullptr;
    for (const Model *m : models)
    {
        if (m->getId() == request->getModel().getId())
        {
            targetModel = m;
            break;
        }
    }
    nonSubmittedRequests.push_back(request->clone(*targetModel));
}

for (const auto &request : other.waitingRequests)
{
    const Model *targetModel = nullptr;
    for (const Model *m : models)
    {
        if (m->getId() == request->getModel().getId())
        {
            targetModel = m;
            break;
        }
    }
    waitingRequests.push_back(request->clone(*targetModel));
}

for (const auto &request : other.activeRequests)
{
    const Model *targetModel = nullptr;
    for (const Model *m : models)
    {
        if (m->getId() == request->getModel().getId())
        {
            targetModel = m;
            break;
        }
    }
    activeRequests.push_back(request->clone(*targetModel));
}
    }
    return *this;
}

Server::Server(Server &&other) noexcept
    : gpus(std::move(other.gpus)), models(std::move(other.models)),
      nonSubmittedRequests(std::move(other.nonSubmittedRequests)),
      waitingRequests(std::move(other.waitingRequests)), activeRequests(std::move(other.activeRequests)),
      policy(other.policy), currentCycle(other.currentCycle)
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
        currentCycle = other.currentCycle;
    }
    return *this;
}