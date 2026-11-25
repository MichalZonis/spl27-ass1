#include "Request.h"
#include "Server.h"
#include "Model.h"

Request::Request(int id, const Model &model, int admissionCycle, std::string prompt)
    : id(id), model(model), admissionCycle(admissionCycle), tokens(model.tokenizer(prompt)),
      reservedBlocks(0), theChoosenGPU(-1), processedTokens(0)
{
}
GenerativeRequest::GenerativeRequest(int id, const Model &model, int admissionCycle, std::string prompt, std::string scriptedOutput,
                                     std::size_t maxOutputTokens)
    : Request(id, model, admissionCycle, prompt), scriptedOutput(scriptedOutput), maxOutputTokens(maxOutputTokens)
{
}
EmbeddingRequest::EmbeddingRequest(int id, const Model &model, int admissionCycle, std::string prompt, int runningSum)
    : Request(id, model, admissionCycle, prompt), runningSum(runningSum)
{
}

int Request::getAdmissionCycle() const
{
    return admissionCycle;
}
int Request::getId() const
{
    return id;
}
const Model &Request::getModel() const
{
    return model;
}

bool EmbeddingRequest::finished() const
{
    return processedTokens == tokens.size();
}
bool GenerativeRequest::finished() const
{
    return processedTokens == tokens.size();
}
void GenerativeRequest::act(GPU &gpu)
{
    if (finished())
        return;
    std::size_t slot;

    if (!memoryBlocks.empty())
        slot = memoryBlocks.back();

    std::size_t t = processedTokens % (gpu.getBlockSize() / model.getEmbeddingSize());
    if (t == 0)
    {
        slot = gpu.allocateBlock(*this);
        gpu.acquire(slot);
        memoryBlocks.push_back(slot);
        reservedBlocks--;
    }
    model.generateTokenEmbedding(tokens, gpu, slot, t * model.getEmbeddingSize());
    processedTokens++;
    t = processedTokens % (gpu.getBlockSize() / model.getEmbeddingSize());
    if (t == 0)
        gpu.assignPrefixToBlock(slot, tokens, model.getId());
}
void EmbeddingRequest::act(GPU &gpu)
{
    if (finished())
        return;
    std::size_t slot;
    if (!memoryBlocks.empty())
        slot = memoryBlocks.back();
    else
    {
        slot = gpu.allocateBlock(*this);
        gpu.acquire(slot);
        memoryBlocks.push_back(slot);
        reservedBlocks--;
    }
    model.generateTokenEmbedding(tokens, gpu, slot, 0);
    runningSum += gpu.readFromBlock(slot, 0);
    processedTokens++;
}
void GenerativeRequest::reserveMemory(GPU &gpu)
{
    std::size_t neededBlocks = gpu.getBlocksForMemory(getNeededMemory(gpu));
    gpu.reserve(neededBlocks, *this);
    reservedBlocks = neededBlocks;
}
void EmbeddingRequest::reserveMemory(GPU &gpu)
{
    std::size_t neededBlocks = gpu.getBlocksForMemory(getNeededMemory(gpu));
    gpu.reserve(neededBlocks, *this);
    reservedBlocks = neededBlocks;
}
void Request::deallocateMemory(GPU &gpu)
{
    gpu.unreserve(getNeedToUnreserve(), *this);
    reservedBlocks = 0;
    for (std::size_t blockId : memoryBlocks)
    {
        gpu.release(blockId);
    }
}
std::size_t GenerativeRequest::getNeedToUnreserve() const
{
    return reservedBlocks;
}
std::size_t EmbeddingRequest::getNeedToUnreserve() const
{
    return reservedBlocks;
}
std::string GenerativeRequest::getOutput() const
{
    return (scriptedOutput == "") ? scriptedOutput : " " + scriptedOutput;
}
std::string EmbeddingRequest::getOutput() const
{
    return std::to_string(runningSum / tokens.size());
}
std::size_t GenerativeRequest::getMaxNeededMemory() const
{
    return model.getEmbeddingSize() * (maxOutputTokens + tokens.size());
}
std::size_t EmbeddingRequest::getMaxNeededMemory() const
{
    return model.getEmbeddingSize() * 2;
}
std::size_t GenerativeRequest::getNeededMemory(const GPU &gpu) const
{
    return model.getEmbeddingSize() * (maxOutputTokens + tokens.size() - gpu.findLongestCachedPrefix(model.getId(), model.getEmbeddingSize(), tokens).matchedTokens);
}
std::size_t EmbeddingRequest::getNeededMemory(const GPU &gpu) const
{
    return model.getEmbeddingSize() * 2;
}
std::vector<std::string> GenerativeRequest::getCacheableTokens() const
{
    return tokens;
}
void GenerativeRequest::allocateOneBlock(GPU &gpu, std::size_t slot) const
{
    gpu.addMemoryBlock(*this, slot);
}
void EmbeddingRequest::allocateOneBlock(GPU &gpu, std::size_t slot) const
{
    gpu.addMemoryBlock(*this, slot);
}
std::vector<std::string> EmbeddingRequest::getCacheableTokens() const
{
    return std::vector<std::string>();
}
Request *GenerativeRequest::clone(const Model &model) const
{
    auto newRequest = new GenerativeRequest(id, model, admissionCycle, "", scriptedOutput, maxOutputTokens);
    newRequest->tokens = tokens;
    newRequest->memoryBlocks = memoryBlocks;
    newRequest->reservedBlocks = reservedBlocks;
    return newRequest;
}
Request *EmbeddingRequest::clone(const Model &model) const
{
    auto newRequest = new EmbeddingRequest(id, model, admissionCycle, "", runningSum);
    newRequest->tokens = tokens;
    newRequest->memoryBlocks = memoryBlocks;
    newRequest->reservedBlocks = reservedBlocks;
    return newRequest;
}

GenerativeRequest::~GenerativeRequest()
{
}

EmbeddingRequest::~EmbeddingRequest()
{
}

Request::~Request() {}