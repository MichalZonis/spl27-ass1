#include "GPU.h"
#include "MemoryBlock.h"
#include <stdexcept>
#include <algorithm>

GPU::GPU(std::size_t blocksNum, std::size_t blockSize)
    : blocksNum(blocksNum),
      blockSize(blockSize),
      memory(blocksNum, nullptr),
      activeBlocksCounter(0),
      reservedBlocksCounter(0),
      cachedPrefixes(blocksNum, nullptr),
      cachedModelIds(blocksNum, ""),
      requestsReservations(),
      evictionCounter(0),
      cacheHits(0)
{
}

GPU::GPU(const GPU &other)
    : blocksNum(other.blocksNum),
      blockSize(other.blockSize),
      memory(other.blocksNum, nullptr),
      activeBlocksCounter(other.activeBlocksCounter),
      reservedBlocksCounter(other.reservedBlocksCounter),
      cachedPrefixes(other.blocksNum, nullptr),
      cachedModelIds(other.blocksNum, ""),
      requestsReservations(other.requestsReservations),
      evictionCounter(other.evictionCounter),
      cacheHits(other.cacheHits)
{

    for (std::size_t i = 0; i < blocksNum; ++i)
    {
        if (other.memory[i] != nullptr)
        {
            memory[i] = other.memory[i]->clone();
            for (std::size_t j = 0; j < other.memory[i]->getRefCount(); j++)
            {
                memory[i]->acquire();
            }
        }
        if (other.cachedPrefixes[i] != nullptr)
        {
            cachedPrefixes[i] = new std::vector<std::string>(*other.cachedPrefixes[i]);
        }
        cachedModelIds[i] = other.cachedModelIds[i];
    }
}

GPU::GPU(GPU &&other)
    : blocksNum(other.blocksNum),
      blockSize(other.blockSize),
      memory(other.blocksNum, nullptr),
      activeBlocksCounter(other.activeBlocksCounter),
      reservedBlocksCounter(other.reservedBlocksCounter),
      cachedPrefixes(other.blocksNum, nullptr),
      cachedModelIds(other.blocksNum, ""),
      requestsReservations(std::move(other.requestsReservations)),
      evictionCounter(other.evictionCounter),
      cacheHits(other.cacheHits)
{

    for (std::size_t i = 0; i < blocksNum; ++i)
    {
        if (other.memory[i] != nullptr)
        {
            memory[i] = other.memory[i];
            other.memory[i] = nullptr;
        }
        if (other.cachedPrefixes[i] != nullptr)
        {
            cachedPrefixes[i] = other.cachedPrefixes[i];
        }
        cachedModelIds[i] = other.cachedModelIds[i];
        other.cachedPrefixes[i] = nullptr;
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
    return findAvailableSlot({});
}

void GPU::evict(std::size_t slot)
{
    evictionCounter++;
    delete memory[slot];
    memory[slot] = nullptr;
    cachedPrefixes[slot] = nullptr;
    cachedModelIds[slot] = "";
}

std::size_t GPU::findAvailableSlot(const std::vector<std::size_t> &blocksToIgnore)
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
        if (std::find(blocksToIgnore.begin(), blocksToIgnore.end(), i) == blocksToIgnore.end() &&
            memory[i] != nullptr && memory[i]->getRefCount() == 0)
        {
            evict(i);
            return i;
        }
    }
    throw std::runtime_error("No available slots in GPU memory.");
}

void GPU::assignPrefixToBlock(std::size_t block,
                              const std::vector<std::string> &prefixTokens, const std::string &modelId)
{
    int exisitingBlock = -1;
    for (size_t i = 0; i < blocksNum; i++)
    {
        if (cachedModelIds[i] == modelId && cachedPrefixes[i] != nullptr && *cachedPrefixes[i] == prefixTokens)
        {
            exisitingBlock = i;
            break;
        }
    }

    if (exisitingBlock == -1)
    {
        cachedModelIds[block] = modelId;
        cachedPrefixes[block] = new std::vector<std::string>(prefixTokens);
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
                // Check if the cached prefix matches the corresponding part of the prompt 
                //(and by extension, the already matched prefix)
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
    std::vector<std::size_t> protectedSlots;
    std::vector<std::size_t> blocksToCopy;

    // 1. Find duplicates (protect their slots) and collect only missing blocks
    for (const std::size_t blockId : blocks)
    {
        if (other.cachedPrefixes[blockId] == nullptr)
        {
            continue;
        }

        const std::vector<std::string> &prefixCopy = *other.cachedPrefixes[blockId];
        const std::string &modelIdCopy = other.cachedModelIds[blockId];
        bool isDuplicate = false;

        for (std::size_t i = 0; i < blocksNum; ++i)
        {
            if (cachedPrefixes[i] != nullptr &&
                cachedModelIds[i] == modelIdCopy &&
                *cachedPrefixes[i] == prefixCopy)
            {
                isDuplicate = true;
                protectedSlots.push_back(i);
                break;
            }
        }

        if (!isDuplicate)
        {
            blocksToCopy.push_back(blockId);
        }
    }

    // 2. Deep-copy missing blocks and protect each newly filled slot
    for (const std::size_t blockId : blocksToCopy)
    {
        std::size_t slot = findAvailableSlot(protectedSlots);
        protectedSlots.push_back(slot);

        cachedModelIds[slot] = other.cachedModelIds[blockId];
        cachedPrefixes[slot] = new std::vector<std::string>(*other.cachedPrefixes[blockId]);
        memory[slot] = other.memory[blockId]->clone();
    }
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
    cacheHits++;
}

GPU &GPU::operator=(GPU &&other)
{

    if (this == &other)
    {
        return *this;
    }

    clear();
    blocksNum = other.blocksNum;
    blockSize = other.blockSize;
    cachedModelIds.reserve(blocksNum);
    cachedModelIds.reserve(blocksNum);
    evictionCounter = other.evictionCounter;
    activeBlocksCounter = other.activeBlocksCounter;
    reservedBlocksCounter = other.reservedBlocksCounter;
    cacheHits = other.cacheHits;
    for (std::size_t i = 0; i < blocksNum; ++i)
    {
        if (other.memory[i] != nullptr)
        {
            memory[i] = other.memory[i];
            other.memory[i] = nullptr;
        }
        else
        {
            memory[i] = nullptr;
        }

        if (other.cachedPrefixes[i] != nullptr)
        {
            cachedPrefixes[i] = other.cachedPrefixes[i];
            other.cachedPrefixes[i] = nullptr;
        }
        else
        {
            cachedPrefixes[i] = nullptr;
        }
        cachedModelIds[i] = other.cachedModelIds[i];
    }

    return *this;
}

GPU &GPU::operator=(const GPU &other)
{
    if (this == &other)
    {
        return *this;
    }

    clear();
    blocksNum = other.blocksNum;
    blockSize = other.blockSize;
    requestsReservations = other.requestsReservations;
    cachedModelIds.reserve(blocksNum);
    cachedModelIds.reserve(blocksNum);
    evictionCounter = other.evictionCounter;
    activeBlocksCounter = other.activeBlocksCounter;
    reservedBlocksCounter = other.reservedBlocksCounter;
    cacheHits = other.cacheHits;
    for (std::size_t i = 0; i < blocksNum; ++i)
    {
        if (other.memory[i] != nullptr)
        {
            memory[i] = other.memory[i]->clone();
            for (std::size_t j = 0; j < other.memory[j]->getRefCount(); j++)
            {
                memory[i]->acquire();
            }
        }
        else
        {
            memory[i] = nullptr;
        }

        if (other.cachedPrefixes[i] != nullptr)
        {
            cachedPrefixes[i] = new std::vector<std::string>(*other.cachedPrefixes[i]);
        }
        else
        {
            cachedPrefixes[i] = nullptr;
        }
        cachedModelIds[i] = other.cachedModelIds[i];
    }

    return *this;
}

std::size_t GPU::getCacheHits() const
{
    return cacheHits;
}

std::size_t GPU::getEvictions() const
{
    return evictionCounter;
}

GPU::~GPU()
{
    clear();
}

void GPU::clear()
{
    for (std::size_t i = 0; i < cachedPrefixes.size(); ++i)
    {
        delete cachedPrefixes[i];
        cachedPrefixes[i] = nullptr;
    }

    for (std::size_t i = 0; i < memory.size(); ++i)
    {
        delete memory[i];
        memory[i] = nullptr;
    }
}

std::size_t GPU::getBlockSize() const { return blockSize; }

std::size_t GPU::countIdles(PrefixMatch prefix) const
{
    std::size_t idleCounter(0);
    for (auto slot : prefix.blockIds)
    {
        if (memory[slot]->getRefCount() == 0)
        {
            idleCounter++;
        }
    }

    return idleCounter;
}
