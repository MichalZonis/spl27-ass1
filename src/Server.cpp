#include <fstream>
#include <stdexcept>
#include "json.hpp"

#include "Server.h"
#include <iostream>

using namespace std;
using json = nlohmann::json;

Config Server::readConfig(string configPath) const{
    std::ifstream input(configPath);
    json config;
    input >> config;
    // All the GPUs have the same block size.
    size_t blockSize = config["block_size"].get<size_t>();
    vector<GPUConfig> gpuConfigs;
    for(auto& gpu: config["gpus"]){
        size_t memoryBytes = gpu["memory_bytes"].get<size_t>();
        GPUConfig gpuConfig{memoryBytes};
        gpuConfigs.push_back(gpuConfig);
    }
    vector<ModelConfig> modelConfigs;
    for(auto& model: config["models"]){
        string id = model["id"].get<string>();
        size_t embeddingSize = model["embedding_size"].get<size_t>();
        ModelConfig modelConfig{id, embeddingSize};
        modelConfigs.push_back(modelConfig);
    }
    vector<RequestConfig> requestConfigs;
    for(auto& request: config["requests"]){
        size_t id = request["id"].get<size_t>();
        string typeStr = request["type"].get<string>();
        if(typeStr != "generative" && typeStr != "text_embedder"){
            throw invalid_argument("invalid request type");
        }
        RequestType type = typeStr=="generative"?RequestType::Generative:RequestType::TextEmbedder;
        string modelId = request["model"].get<string>();
        int admissionCycle = request["cycle"].get<int>();
        string prompt = request["prompt"].get<string>();
        string scriptedOutput = "";
        if (request.count("scripted_output") > 0) {
            scriptedOutput = request["scripted_output"].get<string>();
        }
        size_t maxOutputTokens = 0;
        if (request.count("max_output_tokens") > 0) {
            maxOutputTokens = request["max_output_tokens"].get<size_t>();
        }
        RequestConfig reqConfig{id, type, modelId, admissionCycle, prompt, scriptedOutput, maxOutputTokens};
        requestConfigs.push_back(reqConfig);
    }

    if (config.count("scheduling_policy") == 0) {
        throw invalid_argument("missing scheduling policy");
    }
    string policy = config["scheduling_policy"].get<string>();

    return Config{blockSize, gpuConfigs, modelConfigs, requestConfigs, policy};
}

void Server::serve(){
    while(step());
    std::cout << "All requests have been processed." << std::endl;
}

Server::Server(string configPath)
: gpus(), models(), nonSubmittedRequests(), waitingRequests(), activeRequests(), policy(nullptr){
    Config config = readConfig(configPath);
    gpus.reserve(config.gpus.size());
    for(const auto& gpuConfig: config.gpus){
        gpus.emplace_back(gpuConfig.memoryBytes, config.blockSize);
    }
    models.reserve(config.models.size());
    for(const auto& modelConfig: config.models){
        models.emplace_back(modelConfig.id, modelConfig.embeddingSize);
    }
    //return here after complting policy
    if (config.policy == "arrivalOrder") {
        policy = new ArrivalOrderSchedulingPolicy();
    } else if (config.policy == "shortestJobFirst") {
        policy = new ShortestJobFirstSchedulingPolicy();
    } else if (config.policy == "longestPrefixFirst") {
        policy = new LongestPrefixFirstSchedulingPolicy();
    } else {
        throw invalid_argument("invalid scheduling policy");
    }

    nonSubmittedRequests.reserve(config.requests.size());
    for(const auto& requestConfig: config.requests){
        const Model* model = nullptr;
        for(const auto& m: models){
            if(m.getId() == requestConfig.modelId){
                model = &m;
                break; 
            }
        }
        if(requestConfig.type == RequestType::Generative){
            nonSubmittedRequests.push_back(new GenerativeRequest(requestConfig.id, model, requestConfig.admissionCycle, requestConfig.prompt, requestConfig.scriptedOutput, requestConfig.maxOutputTokens));
        } else if(requestConfig.type == RequestType::TextEmbedder){
            nonSubmittedRequests.push_back(new EmbeddingRequest(requestConfig.id, model, requestConfig.admissionCycle, requestConfig.prompt));
        }
    }

}

Server::~Server() {
        clear();
}
void Server::clear() {
    for (auto request : nonSubmittedRequests) {
        delete request;
    }
    for (auto request : waitingRequests) {
        delete request;
    }
    for (auto request : activeRequests) {
        delete request;
    }
    delete policy;
}
Server::Server(const Server& other)
: gpus(other.gpus), models(other.models), nonSubmittedRequests(), waitingRequests(), activeRequests(), policy(other.policy->clone()) {
    for (const auto& request : other.nonSubmittedRequests) {
        nonSubmittedRequests.push_back(request->clone(*request->getModel()));
    }
    for (const auto& request : other.waitingRequests) 