#ifndef GPU_H_
#define GPU_H_

#include <string>
#include <cstddef>
#include <vector>
#include "Request.h"
#include <unordered_map>

struct PrefixMatch
{
    std::size_t matchedTokens;         // number of prompt tokens
    std::vector<std::size_t> blockIds; // the slots of the cached blocks of the model that cover the longest prefix of the prompt

    PrefixMatch(std::size_t matchedTokens = 0,
                const std::vector<std::size_t> &blockIds = {})
        : matchedTokens(matchedTokens), blockIds(blockIds) {}
};

class MemoryBlock;

class GPU
{
public:
    GPU(std::size_t blocksNum, std::size_t blockSize);
    std::size_t availableSlots() const;
    std::size_t getBlocksForMemory(std::size_t memory) const;
    void reserve(std::size_t n, const Request &request);
    void unreserve(std::size_t n, const Request &request);
    std::size_t allocateBlock(const Request &request);
    void addMemoryBlock(const GenerativeRequest &request, std::size_t slot);
    void addMemoryBlock(const EmbeddingRequest &request, std::size_t slot);
    void acquire(std::size_t slot);
    void release(std::size_t slot);
    void copyMemoryFromOtherGPU(const GPU &other, const std::vector<std::size_t> &blocks);
    void assignPrefixToBlock(std::size_t block, const std::vector<std::string> &prefixTokens, const std::string &modelId);
    int readFromBlock(std::size_t blockId, std::size_t offset) const;
    void writeToBlock(std::size_t blockId, std::size_t offset, int value);
    PrefixMatch findLongestCachedPrefix(const std::string &modelId, std::size_t embeddingSize,
                                        const std::vector<std::string> &promptTokens) const;
    ~GPU();
    std::size_t findAvailableSlot();
    std::size_t findAvailableSlot(const std::vector<std::size_t> &blocksToIgnore);
    std::size_t getBlockSize() const;
    void evict(std::size_t slot);
    std::size_t getEvictions() const;
    std::size_t getCacheHits() const;
    GPU &operator=(GPU &&other);
    GPU &operator=(const GPU &other);
    GPU(const GPU &other);
    GPU(GPU &&other);
    std::size_t countIdles(PrefixMatch prefix);

private:
    std::size_t blocksNum;
    std::size_t blockSize;
    std::vector<MemoryBlock *> memory;
    //--------
    std::size_t activeBlocksCounter;
    std::size_t reservedBlocksCounter;
    std::vector<std::vector<std::string> *> cachedPrefixes;
    std::vector<std::string> cachedModelIds;                   // consider switching this and cachedPrefixes to a map of slot to <modelId, prefix> pair
    std::unordered_map<int, std::size_t> requestsReservations; //{request.getId(), reservedBlocksNum}
    std::size_t evictionCounter;
    void decreaseReservations(const Request &request);
    std::size_t cacheHits;
    void clear();
};

#endif