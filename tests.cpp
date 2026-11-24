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

#define ASSERT_TRUE(cond)                                         \
    do                                                            \
    {                                                             \
        if (!(cond))                                              \
        {                                                         \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ \
                      << " Expected " << #cond << std::endl;      \
            ++g_failedTests;                                      \
        }                                                         \
    } while (0)

#define ASSERT_EQ(a, b)                                                   \
    do                                                                    \
    {                                                                     \
        if (!((a) == (b)))                                                \
        {                                                                 \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__         \
                      << " Expected " << #a << " == " << #b << std::endl; \
            ++g_failedTests;                                              \
        }                                                                 \
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
// 1. Model Tests
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
    std::cout << "  [PASS] testModel\n";
}

// ============================================================================
// 2. MemoryBlock, KVBlock, EmbeddingWorkspace & Rule of 5 Tests
// ============================================================================
void testMemoryBlocks()
{
    std::cout << "Running testMemoryBlocks...\n";

    // Zero initialization & read/write
    KVBlock kv(4);
    for (std::size_t i = 0; i < 4; ++i)
    {
        ASSERT_EQ(kv.readFromBlock(i), 0);
    }
    kv.writeToBlock(2, 42);
    ASSERT_EQ(kv.readFromBlock(2), 42);

    // acquire / release / onLastRelease
    kv.acquire();
    kv.acquire();
    // First release: refCount 2 -> 1 (not last release)
    kv.release();
    // Second release: refCount 1 -> 0 -> KeepCached
    ASSERT_EQ(kv.release(), IdleAction::KeepCached);

    EmbeddingWorkspace ew(4);
    ew.acquire();
    ASSERT_EQ(ew.release(), IdleAction::Destroy);

    // Deep copy constructor (must reset refCount to 0 and own separate array)
    kv.acquire();
    KVBlock kvCopy(kv);
    ASSERT_EQ(kvCopy.readFromBlock(2), 42);
    kvCopy.writeToBlock(2, 99);
    ASSERT_EQ(kv.readFromBlock(2), 42); // Original untouched

    // Copy assignment & self-assignment
    KVBlock kvAssigned(2);
    kvAssigned = kv;
    ASSERT_EQ(kvAssigned.readFromBlock(2), 42);
    kvAssigned.writeToBlock(2, 77);
    ASSERT_EQ(kv.readFromBlock(2), 42);
    kvAssigned = kvAssigned; // Self-assignment safety
    ASSERT_EQ(kvAssigned.readFromBlock(2), 77);

    // Polymorphic clone() preserves concrete type and deep-copies data
    MemoryBlock *basePtr = &kv;
    MemoryBlock *clonedKV = basePtr->clone();
    ASSERT_EQ(clonedKV->readFromBlock(2), 42);
    ASSERT_EQ(clonedKV->onLastRelease(), IdleAction::KeepCached);
    clonedKV->writeToBlock(2, -1);
    ASSERT_EQ(kv.readFromBlock(2), 42);
    delete clonedKV; // Tests virtual destructor

    MemoryBlock *clonedEW = ew.clone();
    ASSERT_EQ(clonedEW->onLastRelease(), IdleAction::Destroy);
    delete clonedEW;

    // Move constructor & move assignment (moved-from must be safe to destroy)
    KVBlock kvMoved(std::move(kvCopy));
    ASSERT_EQ(kvMoved.readFromBlock(2), 99);

    KVBlock kvMoveAssigned(2);
    kvMoveAssigned = std::move(kvMoved);
    ASSERT_EQ(kvMoveAssigned.readFromBlock(2), 99);

    kv.release();
    std::cout << "  [PASS] testMemoryBlocks\n";
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

    // GPU Rule of 5: copy preserves active blocks, refCounts, and cached prefixes
    GPU gpuCopy(gpu);
    ASSERT_EQ(gpuCopy.availableSlots(), gpu.availableSlots());
    gpu.release(es0);
    gpu.release(es1);
    // Releasing on original must not affect copy
    ASSERT_EQ(gpuCopy.availableSlots(), static_cast<std::size_t>(1));

    // Move GPU
    GPU gpuMoved(std::move(gpuCopy));
    ASSERT_EQ(gpuMoved.availableSlots(), static_cast<std::size_t>(1));

    std::cout << "  [PASS] testGPU\n";
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

    // Clone test: points to new model instance
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

    std::cout << "  [PASS] testRequests\n";
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

    // clone() test
    SchedulingPolicy *clonedPolicy = longest.clone();
    ASSERT_EQ(clonedPolicy->choose(eligible, gpus), expLongest);
    delete clonedPolicy;

    std::cout << "  [PASS] testSchedulingPolicies\n";
}

// ============================================================================
// 6. Server End-to-End, Rejection, Cross-GPU Copy & Mid-Run Rule of 5 Tests
// ============================================================================
void testServerAndRuleOf5()
{
    std::cout << "Running testServerAndRuleOf5...\n";

    const std::string cfgPath = "/tmp/spl_unit_test_config.json";
    {
        std::ofstream out(cfgPath);
        out << R"({
            "block_size": 4,
            "scheduling_policy": "arrivalOrder",
            "gpus": [
                {"memory_bytes": 12},
                {"memory_bytes": 12}
            ],
            "models": [
                {"id": "llama", "embedding_size": 2},
                {"id": "e5", "embedding_size": 2}
            ],
            "requests": [
                {
                    "id": 1,
                    "type": "generative",
                    "model": "llama",
                    "cycle": 0,
                    "prompt": "a b c",
                    "scripted_output": "x y",
                    "max_output_tokens": 2
                },
                {
                    "id": 2,
                    "type": "generative",
                    "model": "llama",
                    "cycle": 0,
                    "prompt": "too big prompt for three slots",
                    "scripted_output": "z",
                    "max_output_tokens": 5
                },
                {
                    "id": 3,
                    "type": "generative",
                    "model": "llama",
                    "cycle": 2,
                    "prompt": "a b c x",
                    "scripted_output": "ok",
                    "max_output_tokens": 1
                },
                {
                    "id": 4,
                    "type": "embedding",
                    "model": "e5",
                    "cycle": 1,
                    "prompt": "hello world"
                }
            ]
        })";
    }

    // --- Mid-run Copy Test ---
    Server original(cfgPath);

    std::string prefixOut;
    {
        CoutCapture cap;
        ASSERT_TRUE(original.step()); // cycle 0 (request 2 rejected, request 1 admitted & acts)
        ASSERT_TRUE(original.step()); // cycle 1 (request 4 admitted & acts)
        prefixOut = cap.str();
    }
    ASSERT_TRUE(prefixOut.find("rejected 2 cycle=0") != std::string::npos);

    // Copy server right in the middle of execution
    Server copied(original);

    std::string origRest, copyRest;
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
    // Both must produce identical output independently
    ASSERT_EQ(origRest, copyRest);
    ASSERT_TRUE(origRest.find("finished 1") != std::string::npos);
    ASSERT_TRUE(origRest.find("finished 3") != std::string::npos);
    ASSERT_TRUE(origRest.find("finished 4") != std::string::npos);
    ASSERT_TRUE(origRest.find("summary:") != std::string::npos);

    // --- Move Semantics Test ---
    Server src(cfgPath);
    ASSERT_TRUE(src.step());
    Server dst(std::move(src));

    std::string movedFromOut;
    {
        CoutCapture cap;
        src.serve();
        movedFromOut = cap.str();
    }
    ASSERT_EQ(movedFromOut, "summary: cycles=0 cache_hits=0 evictions=0\n");

    {
        CoutCapture cap;
        dst.serve();
        ASSERT_TRUE(cap.str().find("summary:") != std::string::npos);
    }

    std::remove(cfgPath.c_str());
    std::cout << "  [PASS] testServerAndRuleOf5\n";
}

int main()
{
    testModel();
    testMemoryBlocks();
    testGPU();
    testRequests();
    testSchedulingPolicies();
    testServerAndRuleOf5();
    // std::cout << "\nALL UNIT TESTS PASSED!\n";
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
    // return 0;
}