#include "GPU.h"
#include "MemoryBlock.h"

GPU::GPU(std::size_t blocksNum, std::size_t blockSize)
    : blocksNum(blocksNum), blockSize(blockSize), memory(blocksNum, nullptr), cachedPrefixes(blocksNum, nullptr), cachedModelIds(blocksNum, "") {}

GPU::GPU(const GPU &other)
{
    blocksNum = other.blocksNum;
    blockSize = other.blockSize;
    memory.resize(blocksNum, nullptr);
    cachedPrefixes.resize(blocksNum, nullptr);
    cachedModelIds.resize(blocksNum, "");

    for (std::size_t i = 0; i < blocksNum; ++i)
    {
        if (other.memory[i] != nullptr)
        {
            memory[i] = other.memory[i]->clone();
        }
        if (other.cachedPrefixes[i] != nullptr)
        {
            cachedPrefixes[i] = new PrefixMatch(*other.cachedPrefixes[i]);
        }
        cachedModelIds[i] = other.cachedModelIds[i];
    }
}

std::size_t GPU::availableSlots() const
{
    return blocksNum - activeBlocks() - reservedBlocks.size();
    // TODO: reservedBlocks use is still unclear -
    //  does it keep the blocks reserved for a request, or is it just a count of how many blocks are reserved?
}

std::size_t GPU::activeBlocks() const
{
    // TODO: this might be needed to be updated per reserve/unreserve and such
    std::size_t count = 0;
    for (const auto &block : memory)
    {
        if (block != nullptr && block->getRefCount() > 0)
        {
            count++;
        }
    }
    return count;
}

std::size_t GPU::getBlocksForMemory(std::size_t memory) const
{
    return memory / blockSize + (memory % blockSize != 0);
}

void GPU::reserve(std::size_t n, const Request &request)
{
    // TODO: Implementation for reserving blocks
}

void GPU::unreserve(std::size_t n, const Request &request)
{
    // TODO: Implementation for unreserving blocks
}

void assignPrefixToBlock(std::size_t block,
                         const std::vector<std::string> &prefixTokens, const std::string &modelId)
{
}

PrefixMatch GPU::findLongestCachedPrefix(const std::string &modelId,
                                         std::size_t embeddingSize,
                                         const std::vector<std::string> &promptTokens) const
{
}

void GPU::release(std::size_t slot)
{
    (*memory[slot]).release();
    if ((*memory[slot]).getRefCount() == 0)
    {
        IdleAction action = (*memory[slot]).onLastRelease();
        if (action == IdleAction::Destroy)
        { // TODO: do Destroy action blocks have a cached prefix? if so it needs to be removed from cachedPrefixes as well
            delete memory[slot];
            memory[slot] = nullptr;
        }
        else if (action == IdleAction::KeepCached && cachedPrefixes[slot] == nullptr)
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

GPU::~GPU()
{
    for (auto block : memory)
    {
        delete block;
    }
}
