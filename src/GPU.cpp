#include "GPU.h"
#include "MemoryBlock.h"
#include <stdexcept>
#include <algorithm>

GPU::GPU(std::size_t blocksNum, std::size_t blockSize)
    : blocksNum(blocksNum),
      blockSize(blockSize),
      memory(blocksNum, nullptr),
      cachedPrefixes(blocksNum, nullptr),
      cachedModelIds(blocksNum, ""),
      evictionCounter(0)
{
}

GPU::GPU(const GPU &other)
    : blocksNum(other.blocksNum),
      blockSize(other.blockSize),
      memory(other.blocksNum, nullptr),
      cachedPrefixes(other.blocksNum, nullptr),
      cachedModelIds(other.blocksNum, ""),
      evictionCounter(other.evictionCounter)
{

    for (std::size_t i = 0; i < blocksNum; ++i)
    {
        if (other.memory[i] != nullptr)
        {
            memory[i] = other.memory[i]->clone();
        }
        if (other.cachedPrefixes[i] != nullptr)
        {
            cachedPrefixes[i] = other.cachedPrefixes[i]; // TODO: Consider deep copying the cached prefixes if necessary
        }
        cachedModelIds[i] = other.cachedModelIds[i];
    }
}

std::size_t GPU::availableSlots() const
{
    return blocksNum - activeBlocksCounter - reservedBlocksCounter;
}

std::size_t GPU::getBlocksForMemory(std::size_t memory) const
{
    return memory / blockSize + (memory % blockSize != 0);
}

void GPU::reserve(std::size_t n, const Request &request)
{
    requestsReservations[request.getId()] += n;
    reservedBlocksCounter += n;
}

void GPU::unreserve(std::size_t n, const Request &request)
{
    if (requestsReservations[request.getId()] < n)
    {
        throw std::runtime_error("Request has not reserved enough memory to unreserve.");
    }
    requestsReservations[request.getId()] -= n;
    reservedBlocksCounter -= n;
}

std::size_t GPU::allocateBlock(const Request &request)
{
    if (requestsReservations.find(request.getId()) == requestsReservations.end() ||
        requestsReservations[request.getId()] <= 0)
    {
        throw std::runtime_error("Request has not reserved enough memory for this allocation.");
    }
    else
    {
        decreaseReservations(request);
        std::size_t slot = findAvailableSlot();
        request.allocateOneBlock(*this, slot);
        return slot;
    }
}

void GPU::decreaseReservations(const Request &request)
{
    requestsReservations[request.getId()]--;
    reservedBlocksCounter--;
    if (requestsReservations[request.getId()] == 0)
    {
        requestsReservations.erase(request.getId());
    }
}

void GPU::addMemoryBlock(const GenerativeRequest &request, std::size_t slot)
{
    memory[slot] = new KVBlock(blockSize);
}

void GPU::addMemoryBlock(const EmbeddingRequest &request, std::size_t slot)
{
    memory[slot] = new EmbeddingWorkspace(blockSize);
}

std::size_t GPU::findAvailableSlot()
{

    for (std::size_t i = 0; i < blocksNum; ++i)
    {
        if (memory[i] == nullptr)
        {
            return i;
        }
    }
    for (std::size_t i = 0; i < blocksNum; ++i)
    {
        if (memory[i] != nullptr && memory[i]->getRefCount() == 0)
        {
            evictionCounter++;
            return i;
        }
    }
    throw std::runtime_error("No available slots in GPU memory.");
}

std::size_t GPU::findAvailableSlot(std::vector<std::size_t> &blocksToIgnore)
{

    for (std::size_t i = 0; i < blocksNum; ++i)
    {
        if (std::find(blocksToIgnore.begin(), blocksToIgnore.end(), i) == blocksToIgnore.end() && memory[i] == nullptr)
        {
            return i;
        }
    }
    for (std::size_t i = 0; i < blocksNum; ++i)
    {
        if (std::find(blocksToIgnore.begin(), blocksToIgnore.end(), i) == blocksToIgnore.end() && memory[i] != nullptr && memory[i]->getRefCount() == 0)
        {
            evictionCounter++;
            return i;
        }
    }
    throw std::runtime_error("No available slots in GPU memory.");
}

void GPU::assignPrefixToBlock(std::size_t block,
                              const std::vector<std::string> &prefixTokens, const std::string &modelId)
{
    size_t exisitingBlock = -1;
    for (auto i = 0; i < blocksNum; i++)
    {
        if (cachedModelIds[i] == modelId && cachedPrefixes[i] != nullptr && *cachedPrefixes[i] == prefixTokens)
        {
            exisitingBlock = i;
            break;
        }
    }

    if (exisitingBlock != -1)
    {
        cachedModelIds[block] = modelId;
        cachedPrefixes[block] = new std::vector<std::string>(prefixTokens); // TODO: how is this vector created? if it an rvalue, no need to new it, just assign it.
                                                                            // if it is an lvalue, then we need to copy it.
    }
}

PrefixMatch GPU::findLongestCachedPrefix(const std::string &modelId,
                                         std::size_t embeddingSize,
                                         const std::vector<std::string> &promptTokens) const
{
    std::size_t t = blockSize / embeddingSize;
    std::size_t maxNumOfBlocks = promptTokens.size() / t;
    PrefixMatch result;
    result.matchedTokens = 0;

    bool foundNextBlock = true;
    while (foundNextBlock && result.blockIds.size() < maxNumOfBlocks)
    {
        foundNextBlock = false;
        std::size_t targetLen = result.matchedTokens + t;

        for (std::size_t i = 0; i < blocksNum; i++)
        {
            if (cachedModelIds[i] == modelId &&
                cachedPrefixes[i] != nullptr &&
                cachedPrefixes[i]->size() == targetLen)
            {
                bool blockMatches = true;
                // Check if the cached prefix matches the corresponding part of the prompt (and by extension, the already matched prefix)
                for (std::size_t j = 0; j < targetLen; j++)
                {
                    if ((*cachedPrefixes[i])[j] != promptTokens[j])
                    {
                        blockMatches = false;
                        break;
                    }
                }

                if (blockMatches)
                {
                    result.matchedTokens += t;
                    result.blockIds.push_back(i);
                    foundNextBlock = true;
                    break;
                }
            }
        }
    }

    return result;
}

void GPU::copyMemoryFromOtherGPU(const GPU &other, const std::vector<std::size_t> &blocks)
{
}

void GPU::release(std::size_t slot)
{
    (*memory[slot]).release();
    if ((*memory[slot]).getRefCount() == 0)
    {
        activeBlocksCounter--;
        IdleAction action = (*memory[slot]).onLastRelease();
        if ((action == IdleAction::Destroy) || (action == IdleAction::KeepCached && cachedPrefixes[slot] == nullptr))
        {
            delete memory[slot];
            memory[slot] = nullptr;
        }
    }
}

int GPU::readFromBlock(std::size_t blockId, std::size_t offset) const
{
    return memory[blockId]->readFromBlock(offset);
}

void GPU::writeToBlock(std::size_t blockId, std::size_t offset, int value)
{
    memory[blockId]->writeToBlock(offset, value);
}

void GPU::acquire(std::size_t slot)
{
    (*memory[slot]).acquire();
    activeBlocksCounter++;
}

GPU &GPU::operator=(const GPU &other)
{
    // TODO: Implement assignment operator if needed
}

GPU::~GPU()
{
    for (auto block : memory)
    {
        delete block;
    }
}
