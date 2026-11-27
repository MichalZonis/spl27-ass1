#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Model.h"
#include "MemoryBlock.h"
#include "GPU.h"
#include "Request.h"
#include "SchedulingPolicy.h"
#include "Server.h"

static int g_failedTests = 0;

// Helpers to format values for clear Expected vs. Actual output
template <typename T>
std::string toStr(const T &val)
{
    std::ostringstream oss;
    oss << val;
    return oss.str();
}

inline std::string toStr(const std::string &val)
{
    return "\"" + val + "\"";
}

inline std::string toStr(const char *val)
{
    return std::string("\"") + (val ? val : "nullptr") + "\"";
}

inline std::string toStr(IdleAction action)
{
    switch (action)
    {
    case IdleAction::KeepCached:
        return "IdleAction::KeepCached";
    case IdleAction::Destroy:
        return "IdleAction::Destroy";
    default:
        return "IdleAction(" + std::to_string(static_cast<int>(action)) + ")";
    }
}

template <typename T>
std::string toStr(const std::vector<T> &vec)
{
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < vec.size(); ++i)
    {
        if (i > 0)
            oss << ", ";
        oss << toStr(vec[i]);
    }
    oss << "]";
    return oss.str();
}

#define ASSERT_TRUE(cond)                                           \
    do                                                              \
    {                                                               \
        bool _res = static_cast<bool>(cond);                        \
        if (_res)                                                   \
        {                                                           \
            std::cerr << "  [PASS] " << __FILE__ << ":" << __LINE__ \
                      << " | " << #cond                             \
                      << " | Expected: true | Actual: true\n";      \
        }                                                           \
        else                                                        \
        {                                                           \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__ \
                      << "\n    Condition: " << #cond               \
                      << "\n    Expected:  true"                    \
                      << "\n    Actual:    false" << std::endl;     \
            ++g_failedTests;                                        \
        }                                                           \
    } while (0)

#define ASSERT_EQ(actual, expected)                                           \
    do                                                                        \
    {                                                                         \
        auto _actual = (actual);                                              \
        auto _expected = (expected);                                          \
        if (_actual == _expected)                                             \
        {                                                                     \
            std::cerr << "  [PASS] " << __FILE__ << ":" << __LINE__           \
                      << " | " << #actual                                     \
                      << " | Expected: " << toStr(_expected)                  \
                      << " | Actual: " << toStr(_actual) << "\n";             \
        }                                                                     \
        else                                                                  \
        {                                                                     \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__           \
                      << "\n    Expression: " << #actual                      \
                      << "\n    Expected:   " << toStr(_expected)             \
                      << "\n    Actual:     " << toStr(_actual) << std::endl; \
            ++g_failedTests;                                                  \
        }                                                                     \
    } while (0)

#define ASSERT_CONTAINS(haystack, needle)                                   \
    do                                                                      \
    {                                                                       \
        std::string _h = (haystack);                                        \
        std::string _n = (needle);                                          \
        if (_h.find(_n) != std::string::npos)                               \
        {                                                                   \
            std::cerr << "  [PASS] " << __FILE__ << ":" << __LINE__         \
                      << " | Contains: " << toStr(_n)                       \
                      << " | Actual: " << toStr(_h) << "\n";                \
        }                                                                   \
        else                                                                \
        {                                                                   \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__         \
                      << "\n    Expected output to contain: " << toStr(_n)  \
                      << "\n    Actual output:\n" << _h << std::endl;       \
            ++g_failedTests;                                                \
        }                                                                   \
    } while (0)

// Helper to capture std::cout during Server::serve() / step()
class CoutCapture
{
    std::ostringstream buffer;
    std::streambuf *oldBuf;

public:
    CoutCapture() : buffer(), oldBuf(std::cout.rdbuf(buffer.rdbuf())) {}
    ~CoutCapture() { std::cout.rdbuf(oldBuf); }
    CoutCapture(const CoutCapture &) = delete;
    CoutCapture &operator=(const CoutCapture &) = delete;
    std::string str() const { return buffer.str(); }
};

// ============================================================================
// 1. MemoryBlock, KVBlock, EmbeddingWorkspace & Rule of 5 Tests (Bottom Layer)
// ============================================================================
void testMemoryBlocksRuleOf5()
{
    std::cout << "Running testMemoryBlocksRuleOf5...\n";

    // --- KVBlock basics & refCount ---
    KVBlock kv(4);
    for (std::size_t i = 0; i < 4; ++i)
    {
        ASSERT_EQ(kv.readFromBlock(i), 0);
    }
    kv.writeToBlock(0, 11);
    kv.writeToBlock(2, 42);
    ASSERT_EQ(kv.readFromBlock(2), 42);

    kv.acquire();
    kv.acquire();
    kv.release(); // refCount 2 -> 1
    ASSERT_EQ(kv.release(), IdleAction::KeepCached); // refCount 1 -> 0

    // --- KVBlock Rule of 5 ---
    // 1. Copy constructor (deep copies data & resets refCount to 0)
    kv.acquire();
    KVBlock kvCopy(kv);
    ASSERT_EQ(kvCopy.readFromBlock(0), 11);
    ASSERT_EQ(kvCopy.readFromBlock(2), 42);
    kvCopy.writeToBlock(2, 99);
    ASSERT_EQ(kv.readFromBlock(2), 42); // Original untouched

    // 2. Copy assignment (including differing sizes & self-assignment safety)
    KVBlock kvAssigned(2);
    kvAssigned = kv;
    ASSERT_EQ(kvAssigned.readFromBlock(0), 11);
    ASSERT_EQ(kvAssigned.readFromBlock(2), 42);
    kvAssigned.writeToBlock(2, 77);
    ASSERT_EQ(kv.readFromBlock(2), 42);
    KVBlock &kvSelfRef = kvAssigned;
    kvAssigned = kvSelfRef;
    ASSERT_EQ(kvAssigned.readFromBlock(2), 77);

    // 3. Move constructor (moved-from remains safe to destroy)
    KVBlock kvMoved(std::move(kvCopy));
    ASSERT_EQ(kvMoved.readFromBlock(2), 99);

    // 4. Move assignment (replaces existing buffer cleanly)
    KVBlock kvMoveAssigned(2);
    kvMoveAssigned.writeToBlock(0, -5);
    kvMoveAssigned = std::move(kvMoved);
    ASSERT_EQ(kvMoveAssigned.readFromBlock(2), 99);

    // 5. Polymorphic clone() & virtual destructor via MemoryBlock*
    MemoryBlock *kvBasePtr = &kv;
    MemoryBlock *clonedKV = kvBasePtr->clone();
    ASSERT_EQ(clonedKV->readFromBlock(2), 42);
    ASSERT_EQ(clonedKV->onLastRelease(), IdleAction::KeepCached);
    clonedKV->writeToBlock(2, -1);
    ASSERT_EQ(kv.readFromBlock(2), 42);
    delete clonedKV;

    kv.release();

    // --- EmbeddingWorkspace basics & Rule of 5 ---
    EmbeddingWorkspace ew(4);
    for (std::size_t i = 0; i < 4; ++i)
    {
        ASSERT_EQ(ew.readFromBlock(i), 0);
    }
    ew.writeToBlock(1, 55);
    ew.writeToBlock(3, -12);
    ew.acquire();
    ASSERT_EQ(ew.release(), IdleAction::Destroy);

    // 1. Copy constructor
    ew.acquire();
    EmbeddingWorkspace ewCopy(ew);
    ASSERT_EQ(ewCopy.readFromBlock(1), 55);
    ASSERT_EQ(ewCopy.readFromBlock(3), -12);
    ewCopy.writeToBlock(1, 88);
    ASSERT_EQ(ew.readFromBlock(1), 55);

    // 2. Copy assignment & self-assignment
    EmbeddingWorkspace ewAssigned(2);
    ewAssigned = ew;
    ASSERT_EQ(ewAssigned.readFromBlock(1), 55);
    ASSERT_EQ(ewAssigned.readFromBlock(3), -12);
    ewAssigned.writeToBlock(1, 66);
    ASSERT_EQ(ew.readFromBlock(1), 55);
    EmbeddingWorkspace &ewSelfRef = ewAssigned;
    ewAssigned = ewSelfRef;
    ASSERT_EQ(ewAssigned.readFromBlock(1), 66);

    // 3. Move constructor
    EmbeddingWorkspace ewMoved(std::move(ewCopy));
    ASSERT_EQ(ewMoved.readFromBlock(1), 88);

    // 4. Move assignment
    EmbeddingWorkspace ewMoveAssigned(2);
    ewMoveAssigned.writeToBlock(0, 123);
    ewMoveAssigned = std::move(ewMoved);
    ASSERT_EQ(ewMoveAssigned.readFromBlock(1), 88);

    // 5. Polymorphic clone() & virtual destructor via MemoryBlock*
    MemoryBlock *ewBasePtr = &ew;
    MemoryBlock *clonedEW = ewBasePtr->clone();
    ASSERT_EQ(clonedEW->readFromBlock(1), 55);
    ASSERT_EQ(clonedEW->onLastRelease(), IdleAction::Destroy);
    clonedEW->writeToBlock(1, -9);
    ASSERT_EQ(ew.readFromBlock(1), 55);
    delete clonedEW;

    ew.release();
    std::cout << "  [DONE] testMemoryBlocksRuleOf5\n";
}

// ============================================================================
// 2. Model Tests
// ============================================================================
void testModel()
{
    std::cout << "Running testModel...\n";
    Model m("llama", 4);

    // tokenizer: splits by spaces, strips non-alphanumeric characters
    std::vector<std::string> tokens = m.tokenizer("Hello, world! 123  a-b_c?");
    std::vector<std::string> expected = {"Hello", "world", "123", "abc"};
    ASSERT_EQ(tokens, expected);

    // Pure punctuation token should not leave garbage (or empty tokens if skipped)
    std::vector<std::string> onlyAlnum = m.tokenizer("the cat sat");
    ASSERT_EQ(onlyAlnum.size(), static_cast<std::size_t>(3));
    ASSERT_EQ(onlyAlnum[0], "the");
    ASSERT_EQ(onlyAlnum[2], "sat");

    // generateTokenEmbedding determinism check
    GPU gpu(2, 8);
    GenerativeRequest req(1, m, 0, "the cat", "sat", 1);
    gpu.reserve(2, req);
    std::size_t s0 = gpu.allocateBlock(req);
    std::size_t s1 = gpu.allocateBlock(req);
    gpu.acquire(s0);
    gpu.acquire(s1);

    std::vector<std::string> seq = {"the"};
    m.generateTokenEmbedding(seq, gpu, s0, 0);
    m.generateTokenEmbedding(seq, gpu, s1, 0);
    for (std::size_t i = 0; i < 4; ++i)
    {
        ASSERT_EQ(gpu.readFromBlock(s0, i), gpu.readFromBlock(s1, i));
    }

    gpu.release(s0);
    gpu.release(s1);
    std::cout << "  [DONE] testModel\n";
}

// ============================================================================
// 3. GPU, Prefix Caching, Eviction, Cross-GPU Copy & Rule of 5 Tests
// ============================================================================
void testGPU()
{
    std::cout << "Running testGPU...\n";
    Model llama("llama", 2);
    Model otherModel("mistral", 2);
    // blockSize = 4, embeddingSize = 2 => t = 2 tokens per block
    GPU gpu(3, 4);

    ASSERT_EQ(gpu.availableSlots(), static_cast<std::size_t>(3));
    ASSERT_EQ(gpu.getBlocksForMemory(0), static_cast<std::size_t>(0));
    ASSERT_EQ(gpu.getBlocksForMemory(4), static_cast<std::size_t>(1));
    ASSERT_EQ(gpu.getBlocksForMemory(5), static_cast<std::size_t>(2));

    GenerativeRequest r1(1, llama, 0, "a b c d", "", 0);
    GenerativeRequest r2(2, llama, 0, "a b e f", "", 0);

    // Reserve & unreserve
    gpu.reserve(2, r1);
    ASSERT_EQ(gpu.availableSlots(), static_cast<std::size_t>(1));
    gpu.unreserve(1, r1);
    ASSERT_EQ(gpu.availableSlots(), static_cast<std::size_t>(2));
    gpu.reserve(1, r1); // back to 2 reserved

    // Allocate lowest empty slots (0, 1) and assign prefix keys
    std::size_t s0 = gpu.allocateBlock(r1);
    ASSERT_EQ(s0, static_cast<std::size_t>(0));
    gpu.acquire(s0);
    gpu.writeToBlock(s0, 0, 111);
    gpu.assignPrefixToBlock(s0, {"a", "b"}, "llama");

    std::size_t s1 = gpu.allocateBlock(r1);
    ASSERT_EQ(s1, static_cast<std::size_t>(1));
    gpu.acquire(s1);
    gpu.writeToBlock(s1, 0, 222);
    gpu.assignPrefixToBlock(s1, {"a", "b", "c", "d"}, "llama");

    // 2 active blocks, 0 reserved => 1 available slot
    ASSERT_EQ(gpu.availableSlots(), static_cast<std::size_t>(1));

    // findLongestCachedPrefix: exact chain, model separation, stop at missing block
    PrefixMatch pm = gpu.findLongestCachedPrefix("llama", 2, {"a", "b", "c", "d", "x"});
    ASSERT_EQ(pm.matchedTokens, static_cast<std::size_t>(4));
    ASSERT_EQ(pm.blockIds.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(pm.blockIds[0], s0);
    ASSERT_EQ(pm.blockIds[1], s1);

    PrefixMatch pmWrongModel = gpu.findLongestCachedPrefix("mistral", 2, {"a", "b", "c", "d"});
    ASSERT_EQ(pmWrongModel.matchedTokens, static_cast<std::size_t>(0));
    ASSERT_TRUE(pmWrongModel.blockIds.empty());

    // Duplicate prefix key: second block with same key must NOT be cached
    gpu.reserve(1, r2);
    std::size_t s2 = gpu.allocateBlock(r2);
    ASSERT_EQ(s2, static_cast<std::size_t>(2));
    gpu.acquire(s2);
    gpu.assignPrefixToBlock(s2, {"a", "b"}, "llama"); // duplicate!
    gpu.release(s2);                                  // no prefix key => deleted immediately (slot 2 becomes empty)
    ASSERT_EQ(gpu.availableSlots(), static_cast<std::size_t>(1));

    // Release s0 and s1 => both become idle (cached), availableSlots returns to 3
    gpu.release(s0);
    gpu.release(s1);
    ASSERT_EQ(gpu.availableSlots(), static_cast<std::size_t>(3));

    // copyMemoryFromOtherGPU: copies chain to gpu2 without duplicating existing keys
    GPU gpu2(3, 4);
    gpu2.copyMemoryFromOtherGPU(gpu, {s0, s1});
    PrefixMatch pm2 = gpu2.findLongestCachedPrefix("llama", 2, {"a", "b", "c", "d"});
    ASSERT_EQ(pm2.matchedTokens, static_cast<std::size_t>(4));
    ASSERT_EQ(gpu2.readFromBlock(pm2.blockIds[0], 0), 111);
    ASSERT_EQ(gpu2.readFromBlock(pm2.blockIds[1], 0), 222);

    // Copying again should not duplicate blocks already present on gpu2
    gpu2.copyMemoryFromOtherGPU(gpu, {s0, s1});
    ASSERT_EQ(gpu2.availableSlots(), static_cast<std::size_t>(3));

    // Eviction order: fill slot 2 first, then next allocation evicts lowest idle slot (0)
    EmbeddingRequest e1(3, llama, 0, "hi");
    gpu.reserve(2, e1);
    std::size_t es0 = gpu.allocateBlock(e1);
    ASSERT_EQ(es0, static_cast<std::size_t>(2)); // empty slot first!
    gpu.acquire(es0);
    std::size_t es1 = gpu.allocateBlock(e1);
    ASSERT_EQ(es1, static_cast<std::size_t>(0)); // lowest idle slot evicted!
    gpu.acquire(es1);

    // Since s0 ("a b") was evicted, s1 ("a b c d") alone cannot form a prefix match
    PrefixMatch pmBroken = gpu.findLongestCachedPrefix("llama", 2, {"a", "b", "c", "d"});
    ASSERT_EQ(pmBroken.matchedTokens, static_cast<std::size_t>(0));

    // --- GPU Rule of 5 ---
    // 1. Copy constructor preserves active blocks, refCounts, and cached prefixes
    GPU gpuCopy(gpu);
    ASSERT_EQ(gpuCopy.availableSlots(), gpu.availableSlots());

    // 2. Copy assignment & self-assignment deep-copy blocks independently
    GPU gpuAssigned(1, 2);
    gpuAssigned = gpu;
    ASSERT_EQ(gpuAssigned.availableSlots(), gpu.availableSlots());
    GPU &gpuSelfRef = gpuAssigned;
    gpuAssigned = gpuSelfRef;
    ASSERT_EQ(gpuAssigned.availableSlots(), gpu.availableSlots());

    gpuAssigned.writeToBlock(es0, 0, 999);
    ASSERT_EQ(gpu.readFromBlock(es0, 0), 0);

    // Releasing on original must not affect copy or copy-assigned GPU
    gpu.release(es0);
    gpu.release(es1);
    ASSERT_EQ(gpuCopy.availableSlots(), static_cast<std::size_t>(1));
    ASSERT_EQ(gpuAssigned.availableSlots(), static_cast<std::size_t>(1));

    // 3. Move constructor
    GPU gpuMoved(std::move(gpuCopy));
    ASSERT_EQ(gpuMoved.availableSlots(), static_cast<std::size_t>(1));

    // 4. Move assignment into an already-populated GPU
    GPU gpuMoveAssigned(2, 4);
    gpuMoveAssigned.copyMemoryFromOtherGPU(gpu2, {s0, s1});
    gpuMoveAssigned = std::move(gpuAssigned);
    ASSERT_EQ(gpuMoveAssigned.availableSlots(), static_cast<std::size_t>(1));
    ASSERT_EQ(gpuMoveAssigned.readFromBlock(es0, 0), 999);
    gpuMoveAssigned.release(es0);
    gpuMoveAssigned.release(es1);
    ASSERT_EQ(gpuMoveAssigned.availableSlots(), static_cast<std::size_t>(3));

    std::cout << "  [DONE] testGPU\n";
}

// ============================================================================
// 4. Request (GenerativeRequest & EmbeddingRequest) Tests
// ============================================================================
void testRequests()
{
    std::cout << "Running testRequests...\n";
    Model llama("llama", 2);
    Model e5("e5", 2);
    // blockSize = 4, embeddingSize = 2 => t = 2 tokens/block
    GPU gpu(4, 4);

    // --- GenerativeRequest ---
    // Prompt: "a b c" (n=3), scriptedOutput: "x y" (2 tokens), maxOutputTokens: 4 (m=4)
    // Max needed: 2 * (3 + 4) = 14 ints => 4 blocks
    GenerativeRequest g1(10, llama, 0, "a b c", "x y", 4);
    ASSERT_EQ(g1.getId(), 10);
    ASSERT_EQ(g1.getAdmissionCycle(), 0);
    ASSERT_EQ(g1.getMaxNeededMemory(), static_cast<std::size_t>(14));
    ASSERT_EQ(g1.getNeededMemory(gpu), static_cast<std::size_t>(14));
    ASSERT_EQ(g1.getCacheableTokens().size(), static_cast<std::size_t>(3));

    g1.reserveMemory(gpu);
    ASSERT_EQ(gpu.availableSlots(), static_cast<std::size_t>(0)); // 4 blocks reserved

    // Run until finished: 3 prompt + 2 scripted output = 5 acts
    int acts = 0;
    while (!g1.finished())
    {
        g1.act(gpu);
        ++acts;
    }
    ASSERT_EQ(acts, 5);
    ASSERT_EQ(g1.getOutput(), "x y");
    // Used 3 blocks for 5 tokens (ceil(5/2) = 3), reserved 4 => 1 unused block to unreserve
    ASSERT_EQ(g1.getNeedToUnreserve(), static_cast<std::size_t>(1));

    g1.deallocateMemory(gpu);
    ASSERT_EQ(gpu.availableSlots(), static_cast<std::size_t>(4));

    // Cached blocks left behind by g1: "a b" and "a b c x" (last block "y" was partial => deleted)
    PrefixMatch pm = gpu.findLongestCachedPrefix("llama", 2, {"a", "b", "c", "x", "z"});
    ASSERT_EQ(pm.matchedTokens, static_cast<std::size_t>(4));

    // Second generative request reuses prefix "a b c x" (h = 4)
    GenerativeRequest g2(11, llama, 1, "a b c x z", "w", 1);
    // Needed: 2 * (5 - 4 + 1) = 4 ints => 1 new block
    ASSERT_EQ(g2.getNeededMemory(gpu), static_cast<std::size_t>(4));

    // Polymorphic clone() test for GenerativeRequest: points to new model instance
    Model llamaCopy("llama", 2);
    Request *g2Clone = g2.clone(llamaCopy);
    ASSERT_EQ(g2Clone->getId(), 11);
    ASSERT_TRUE(&g2Clone->getModel() == &llamaCopy);
    delete g2Clone;

    // --- EmbeddingRequest ---
    // Needs 2 * embeddingSize = 4 ints => 1 block regardless of prompt length
    EmbeddingRequest emb(20, e5, 0, "hello world test");
    ASSERT_EQ(emb.getMaxNeededMemory(), static_cast<std::size_t>(4));
    ASSERT_EQ(emb.getNeededMemory(gpu), static_cast<std::size_t>(4));
    ASSERT_TRUE(emb.getCacheableTokens().empty());

    // Polymorphic clone() test for EmbeddingRequest
    Model e5Copy("e5", 2);
    Request *embClone = emb.clone(e5Copy);
    ASSERT_EQ(embClone->getId(), 20);
    ASSERT_TRUE(&embClone->getModel() == &e5Copy);
    delete embClone;

    emb.reserveMemory(gpu);
    ASSERT_EQ(emb.getNeedToUnreserve(), static_cast<std::size_t>(0)); // workspace allocated immediately

    int embActs = 0;
    while (!emb.finished())
    {
        emb.act(gpu);
        ++embActs;
    }
    ASSERT_EQ(embActs, 3);
    std::string embOut = emb.getOutput();
    ASSERT_TRUE(!embOut.empty() && embOut.front() == '[' && embOut.back() == ']');

    emb.deallocateMemory(gpu);
    ASSERT_EQ(gpu.availableSlots(), static_cast<std::size_t>(4));

    std::cout << "  [DONE] testRequests\n";
}

// ============================================================================
// 5. SchedulingPolicy Tests (Exact Example from Assignment Section 4.4.6)
// ============================================================================
void testSchedulingPolicies()
{
    std::cout << "Running testSchedulingPolicies...\n";
    Model llama("llama", 4);
    Model e5("e5", 4);
    // blockSize == embeddingSize => 1 token per block
    GPU gpu(10, 4);

    // Populate GPU cache with "the", "the cat", "the cat sat"
    GenerativeRequest seed(0, llama, 0, "the cat sat", "", 0);
    seed.reserveMemory(gpu);
    while (!seed.finished())
    {
        seed.act(gpu);
    }
    seed.deallocateMemory(gpu);

    std::vector<GPU> gpus;
    gpus.push_back(gpu);

    GenerativeRequest r3(3, llama, 0, "a dog", "", 3);            // blocks=5, h=0
    EmbeddingRequest r4(4, e5, 1, "hi");                          // blocks=2, h=0
    GenerativeRequest r5(5, llama, 2, "the cat ran", "", 1);      // blocks=4, h=2
    GenerativeRequest r6(6, llama, 3, "the cat sat down", "", 4); // blocks=8, h=3

    std::vector<const Request *> eligible = {&r3, &r4, &r5, &r6};

    ArrivalOrderPolicy arrival;
    ShortestJobPolicy shortest;
    LongestPrefixPolicy longest;

    std::vector<int> ordArrival = arrival.choose(eligible, gpus);
    std::vector<int> expArrival = {0, 1, 2, 3}; // ids: 3, 4, 5, 6
    ASSERT_EQ(ordArrival, expArrival);

    std::vector<int> ordShortest = shortest.choose(eligible, gpus);
    std::vector<int> expShortest = {1, 2, 0, 3}; // ids: 4, 5, 3, 6
    ASSERT_EQ(ordShortest, expShortest);

    std::vector<int> ordLongest = longest.choose(eligible, gpus);
    std::vector<int> expLongest = {3, 2, 0, 1}; // ids: 6, 5, 3, 4 (tie 3 vs 4 broken by id)
    ASSERT_EQ(ordLongest, expLongest);

    // Polymorphic clone() tests for all concrete SchedulingPolicy types
    SchedulingPolicy *clonedArrival = arrival.clone();
    ASSERT_EQ(clonedArrival->choose(eligible, gpus), expArrival);
    delete clonedArrival;

    SchedulingPolicy *clonedShortest = shortest.clone();
    ASSERT_EQ(clonedShortest->choose(eligible, gpus), expShortest);
    delete clonedShortest;

    SchedulingPolicy *clonedLongest = longest.clone();
    ASSERT_EQ(clonedLongest->choose(eligible, gpus), expLongest);
    delete clonedLongest;

    std::cout << "  [DONE] testSchedulingPolicies\n";
}

// ============================================================================
// 6. Server End-to-End, Rejection, Prefix Hit & Mid-Run Rule of 5 Tests
// ============================================================================
void testServerAndRuleOf5()
{
    std::cout << "Running testServerAndRuleOf5...\n";

    const std::string cfgPath = "/tmp/spl_unit_test_config.json";
    {
        std::ofstream out(cfgPath);
        out << R"({
            "scheduling_policy": "arrivalOrder",
            "block_size": 8,
            "gpus": [
                { "memory_bytes": 64 },
                { "memory_bytes": 128 }
            ],
            "models": [
                { "id": "gpt", "embedding_size": 4 },
                { "id": "embedder", "embedding_size": 4 }
            ],
            "requests": [
                {
                    "id": 1,
                    "type": "generative",
                    "model": "gpt",
                    "cycle": 0,
                    "prompt": "what is the capital of france",
                    "scripted_output": "paris is the capital",
                    "max_output_tokens": 3
                },
                {
                    "id": 2,
                    "type": "text_embedder",
                    "model": "embedder",
                    "cycle": 1,
                    "prompt": "the dog barked loudly"
                },
                {
                    "id": 3,
                    "type": "generative",
                    "model": "gpt",
                    "cycle": 2,
                    "prompt": "hello world",
                    "scripted_output": "hi there",
                    "max_output_tokens": 5
                },
                {
                    "id": 4,
                    "type": "generative",
                    "model": "gpt",
                    "cycle": 12,
                    "prompt": "what is the capital of italy",
                    "scripted_output": "rome",
                    "max_output_tokens": 2
                },
                {
                    "id": 5,
                    "type": "generative",
                    "model": "gpt",
                    "cycle": 0,
                    "prompt": "oversized request",
                    "scripted_output": "x",
                    "max_output_tokens": 100
                }
            ]
        })";
    }

    // --- Mid-run Copy Constructor & Copy Assignment Test ---
    Server original(cfgPath);

    std::string prefixOut;
    {
        CoutCapture cap;
        ASSERT_TRUE(original.step()); // cycle 0 (request 5 rejected, request 1 admitted)
        ASSERT_TRUE(original.step()); // cycle 1 (request 2 admitted)
        prefixOut = cap.str();
    }
    ASSERT_CONTAINS(prefixOut, "rejected 5 cycle=0");

    // 1. Copy constructor right in the middle of execution
    Server copied(original);

    // 2. Copy assignment & self-assignment onto an already-initialized active Server
    Server copyAssigned(cfgPath);
    {
        CoutCapture cap;
        ASSERT_TRUE(copyAssigned.step());
    }
    copyAssigned = original;
    Server &selfRef = copyAssigned;
    copyAssigned = selfRef;

    std::string origRest, copyRest, copyAssignedRest;
    {
        CoutCapture cap;
        original.serve();
        origRest = cap.str();
    }
    {
        CoutCapture cap;
        copied.serve();
        copyRest = cap.str();
    }
    {
        CoutCapture cap;
        copyAssigned.serve();
        copyAssignedRest = cap.str();
    }

    // All three must produce identical output independently
    ASSERT_EQ(origRest, copyRest);
    ASSERT_EQ(origRest, copyAssignedRest);
    ASSERT_CONTAINS(origRest, "finished 1");
    ASSERT_CONTAINS(origRest, "paris is the"); // truncated to max_output_tokens=3
    ASSERT_CONTAINS(origRest, "finished 2");
    ASSERT_CONTAINS(origRest, "finished 3");
    ASSERT_CONTAINS(origRest, "hi there");
    ASSERT_CONTAINS(origRest, "finished 4");
    ASSERT_CONTAINS(origRest, "rome");
    ASSERT_CONTAINS(origRest, "summary:");

    // --- Mid-run Move Constructor & Move Assignment Test ---
    Server src(cfgPath);
    {
        CoutCapture cap;
        ASSERT_TRUE(src.step());
    }
    // 3. Move constructor
    Server dst(std::move(src));

    std::string movedFromOut;
    {
        CoutCapture cap;
        src.serve();
        movedFromOut = cap.str();
    }
    ASSERT_EQ(movedFromOut, "summary: cycles=0 cache_hits=0 evictions=0\n");

    // 4. Move assignment into a server that already holds active requests and GPU allocations
    Server moveTarget(cfgPath);
    {
        CoutCapture cap;
        ASSERT_TRUE(moveTarget.step());
        ASSERT_TRUE(moveTarget.step());
    }
    moveTarget = std::move(dst);

    std::string dstAfterMoveOut;
    {
        CoutCapture cap;
        dst.serve();
        dstAfterMoveOut = cap.str();
    }
    ASSERT_EQ(dstAfterMoveOut, "summary: cycles=0 cache_hits=0 evictions=0\n");

    {
        CoutCapture cap;
        moveTarget.serve();
        std::string targetOut = cap.str();
        ASSERT_CONTAINS(targetOut, "finished 1");
        ASSERT_CONTAINS(targetOut, "summary:");
    }

    std::remove(cfgPath.c_str());
    std::cout << "  [DONE] testServerAndRuleOf5\n";
}

int main()
{
    testMemoryBlocksRuleOf5();
    testModel();
    testGPU();
    testRequests();
    testSchedulingPolicies();
    testServerAndRuleOf5();
    if (g_failedTests == 0)
    {
        std::cout << "\nALL TESTS PASSED!" << std::endl;
        return 0;
    }
    else
    {
        std::cerr << "\n"
                  << g_failedTests << " ASSERTION(S) FAILED." << std::endl;
        return 1;
    }
}