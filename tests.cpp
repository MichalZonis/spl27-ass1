#include <cassert>
#include <csetjmp>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
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

// ============================================================================
// Test Runner State, Section Isolation & Summary Reporting
// ============================================================================
struct SectionReport
{
    std::string name;
    int passed = 0;
    int failed = 0;
    bool crashed = false;
    bool completed = false;
    std::vector<std::string> failedTests;
};

static std::vector<SectionReport> g_sections;
static int g_activeSectionIdx = -1;
static int g_failedTests = 0;

static const char *g_currentSubtest = "";
static const char *g_lastFile = "tests.cpp";
static int g_lastLine = 0;
static const char *g_lastExpr = "";

static std::streambuf *g_origCoutBuf = nullptr;
static std::jmp_buf g_sectionJmpBuf;
static volatile std::sig_atomic_t g_inSectionGuard = 0;
static volatile std::sig_atomic_t g_caughtSignal = 0;
static volatile std::sig_atomic_t g_summaryPrinted = 0;
static char g_terminateReason[512] = {0};

static const std::vector<std::string> kAllSectionNames = {
    "1. MemoryBlock, KVBlock, EmbeddingWorkspace & Rule of 5",
    "2. Model",
    "3. GPU, Prefix Caching, Eviction, Cross-GPU Copy & Rule of 5",
    "4. Request & Individual Output",
    "5. SchedulingPolicy",
    "6. Server Overall End-to-End Output",
    "7. Server Mid-Run Rule of 5"};

inline const char *signalName(int sig)
{
    switch (sig)
    {
    case SIGSEGV:
        return "SIGSEGV (Segmentation fault)";
    case SIGABRT:
        return "SIGABRT (Aborted / core dump)";
    case SIGFPE:
        return "SIGFPE (Floating point exception)";
    case SIGILL:
        return "SIGILL (Illegal instruction)";
    case SIGBUS:
        return "SIGBUS (Bus error)";
    default:
        return "Unknown fatal signal";
    }
}

void restoreCoutIfNeeded()
{
    if (g_origCoutBuf != nullptr && std::cout.rdbuf() != g_origCoutBuf)
    {
        std::cout.rdbuf(g_origCoutBuf);
    }
}

void printTestSummary()
{
    if (g_summaryPrinted)
    {
        return;
    }
    g_summaryPrinted = 1;

    restoreCoutIfNeeded();
    std::fflush(stdout);
    std::fflush(stderr);

    int totalPassed = 0;
    int totalFailed = 0;

    std::fprintf(stderr, "\n============================================================================\n");
    std::fprintf(stderr, "TEST SUMMARY\n");
    std::fprintf(stderr, "============================================================================\n");

    for (std::size_t i = 0; i < g_sections.size(); ++i)
    {
        const SectionReport &sec = g_sections[i];
        totalPassed += sec.passed;
        totalFailed += sec.failed;

        const char *statusTag = "[PASS]";
        if (sec.crashed)
        {
            statusTag = "[CRASH]";
        }
        else if (sec.failed > 0)
        {
            statusTag = "[FAIL]";
        }

        std::fprintf(stderr, "%s %s: %d passed, %d failed%s\n",
                     statusTag,
                     sec.name.c_str(),
                     sec.passed,
                     sec.failed,
                     sec.crashed ? " (SECTION ABORTED BY ERROR/CRASH)" : "");

        if (!sec.failedTests.empty())
        {
            std::fprintf(stderr, "  Failed tests:\n");
            for (const std::string &failName : sec.failedTests)
            {
                std::fprintf(stderr, "    - %s\n", failName.c_str());
            }
        }
    }

    for (std::size_t i = g_sections.size(); i < kAllSectionNames.size(); ++i)
    {
        std::fprintf(stderr, "[SKIP] %s: not reached\n", kAllSectionNames[i].c_str());
    }

    if (g_terminateReason[0] != '\0')
    {
        std::fprintf(stderr, "Fatal error outside section: %s\n", g_terminateReason);
    }

    std::fprintf(stderr, "----------------------------------------------------------------------------\n");
    std::fprintf(stderr, "Total: %d passed, %d failed\n", totalPassed, totalFailed);
    std::fprintf(stderr, "============================================================================\n");
    std::fflush(stderr);
}

void handleFatalSignal(int sig)
{
    if (g_inSectionGuard)
    {
        g_caughtSignal = sig;
        std::longjmp(g_sectionJmpBuf, 1);
    }

    static volatile std::sig_atomic_t s_inFallbackHandler = 0;
    if (s_inFallbackHandler)
    {
        std::signal(sig, SIG_DFL);
        std::raise(sig);
    }
    s_inFallbackHandler = 1;

    std::snprintf(g_terminateReason, sizeof(g_terminateReason),
                  "Fatal signal %s", signalName(sig));
    printTestSummary();

    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

void handleTerminate()
{
    if (g_inSectionGuard)
    {
        g_caughtSignal = SIGABRT;
        std::longjmp(g_sectionJmpBuf, 1);
    }

    if (g_terminateReason[0] == '\0')
    {
        try
        {
            std::exception_ptr eptr = std::current_exception();
            if (eptr)
            {
                std::rethrow_exception(eptr);
            }
            std::snprintf(g_terminateReason, sizeof(g_terminateReason),
                          "std::terminate called without an active exception");
        }
        catch (const std::exception &e)
        {
            std::snprintf(g_terminateReason, sizeof(g_terminateReason),
                          "Uncaught std::exception: %s", e.what());
        }
        catch (...)
        {
            std::snprintf(g_terminateReason, sizeof(g_terminateReason),
                          "Uncaught unknown non-std exception");
        }
    }

    printTestSummary();
    std::abort();
}

void installSignalHandlers()
{
#if defined(__unix__) || defined(__APPLE__)
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handleFatalSignal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_NODEFER;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGABRT, &sa, nullptr);
    sigaction(SIGFPE, &sa, nullptr);
    sigaction(SIGILL, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
#else
    std::signal(SIGSEGV, handleFatalSignal);
    std::signal(SIGABRT, handleFatalSignal);
    std::signal(SIGFPE, handleFatalSignal);
    std::signal(SIGILL, handleFatalSignal);
    std::signal(SIGBUS, handleFatalSignal);
#endif
}

void installCrashHandlers()
{
    g_origCoutBuf = std::cout.rdbuf();
    g_sections.reserve(kAllSectionNames.size());
    std::atexit(printTestSummary);
    std::set_terminate(handleTerminate);
    installSignalHandlers();
}

void beginSection(const std::string &name)
{
    SectionReport report;
    report.name = name;
    report.failedTests.reserve(32);
    g_sections.push_back(std::move(report));
    g_activeSectionIdx = static_cast<int>(g_sections.size()) - 1;
    g_currentSubtest = "";
    g_lastLine = 0;
    g_lastExpr = "";
    std::cout << "Running " << name << "...\n";
}

void endSection()
{
    if (g_activeSectionIdx >= 0 && static_cast<std::size_t>(g_activeSectionIdx) < g_sections.size())
    {
        g_sections[g_activeSectionIdx].completed = true;
        std::cout << "  [DONE] " << g_sections[g_activeSectionIdx].name << "\n";
    }
}

void recordPass()
{
    if (g_activeSectionIdx >= 0 && static_cast<std::size_t>(g_activeSectionIdx) < g_sections.size())
    {
        ++g_sections[g_activeSectionIdx].passed;
    }
}

void recordFail(const char *file, int line, const std::string &exprDesc)
{
    ++g_failedTests;
    if (g_activeSectionIdx >= 0 && static_cast<std::size_t>(g_activeSectionIdx) < g_sections.size())
    {
        SectionReport &sec = g_sections[g_activeSectionIdx];
        ++sec.failed;
        std::ostringstream oss;
        if (g_currentSubtest && g_currentSubtest[0] != '\0')
        {
            oss << "[" << g_currentSubtest << "] ";
        }
        oss << file << ":" << line << " (" << exprDesc << ")";
        sec.failedTests.push_back(oss.str());
    }
}

void recordSectionAbort(const std::string &reason)
{
    ++g_failedTests;
    if (g_activeSectionIdx >= 0 && static_cast<std::size_t>(g_activeSectionIdx) < g_sections.size())
    {
        SectionReport &sec = g_sections[g_activeSectionIdx];
        sec.crashed = true;
        ++sec.failed;
        std::ostringstream oss;
        if (g_currentSubtest && g_currentSubtest[0] != '\0')
        {
            oss << "[" << g_currentSubtest << "] ";
        }
        oss << reason;
        if (g_lastLine > 0)
        {
            oss << " (at/after " << g_lastFile << ":" << g_lastLine << ": " << g_lastExpr << ")";
        }
        sec.failedTests.push_back(oss.str());
    }
}

void runSectionGuarded(void (*fn)())
{
    g_caughtSignal = 0;
    g_inSectionGuard = 1;

    if (setjmp(g_sectionJmpBuf) == 0)
    {
        try
        {
            fn();
            g_inSectionGuard = 0;
        }
        catch (const std::exception &e)
        {
            g_inSectionGuard = 0;
            restoreCoutIfNeeded();
            std::string msg = std::string("Uncaught std::exception: ") + e.what();
            std::cerr << "  [ERROR] " << msg << " -- continuing to next section...\n";
            recordSectionAbort(msg);
        }
        catch (...)
        {
            g_inSectionGuard = 0;
            restoreCoutIfNeeded();
            std::string msg = "Uncaught unknown non-std exception";
            std::cerr << "  [ERROR] " << msg << " -- continuing to next section...\n";
            recordSectionAbort(msg);
        }
    }
    else
    {
        g_inSectionGuard = 0;
        restoreCoutIfNeeded();
        installSignalHandlers();
        std::string msg = std::string("Fatal signal ") + signalName(static_cast<int>(g_caughtSignal));
        std::cerr << "  [CRASH] " << msg << " -- continuing to next section...\n";
        recordSectionAbort(msg);
    }

    g_activeSectionIdx = -1;
}

#define BEGIN_SECTION(sectionName) beginSection(sectionName)
#define END_SECTION() endSection()
#define SUBTEST(subtestName)              \
    do                                    \
    {                                     \
        g_currentSubtest = (subtestName); \
    } while (0)

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
        g_lastFile = __FILE__;                                      \
        g_lastLine = __LINE__;                                      \
        g_lastExpr = #cond;                                         \
        bool _res = static_cast<bool>(cond);                        \
        if (_res)                                                   \
        {                                                           \
            recordPass();                                           \
            std::cerr << "  [PASS] " << __FILE__ << ":" << __LINE__ \
                      << " | " << #cond                             \
                      << " | Expected: true | Actual: true\n";      \
        }                                                           \
        else                                                        \
        {                                                           \
            recordFail(__FILE__, __LINE__, #cond);                  \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__ \
                      << "\n    Condition: " << #cond               \
                      << "\n    Expected:  true"                    \
                      << "\n    Actual:    false" << std::endl;     \
        }                                                           \
    } while (0)

#define ASSERT_EQ(actual, expected)                                           \
    do                                                                        \
    {                                                                         \
        g_lastFile = __FILE__;                                                \
        g_lastLine = __LINE__;                                                \
        g_lastExpr = #actual " == " #expected;                                \
        auto _actual = (actual);                                              \
        auto _expected = (expected);                                          \
        if (_actual == _expected)                                             \
        {                                                                     \
            recordPass();                                                     \
            std::cerr << "  [PASS] " << __FILE__ << ":" << __LINE__           \
                      << " | " << #actual                                     \
                      << " | Expected: " << toStr(_expected)                  \
                      << " | Actual: " << toStr(_actual) << "\n";             \
        }                                                                     \
        else                                                                  \
        {                                                                     \
            recordFail(__FILE__, __LINE__, #actual " == " #expected);         \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__           \
                      << "\n    Expression: " << #actual                      \
                      << "\n    Expected:   " << toStr(_expected)             \
                      << "\n    Actual:     " << toStr(_actual) << std::endl; \
        }                                                                     \
    } while (0)

#define ASSERT_CONTAINS(haystack, needle)                                   \
    do                                                                      \
    {                                                                       \
        g_lastFile = __FILE__;                                              \
        g_lastLine = __LINE__;                                              \
        g_lastExpr = #haystack " contains " #needle;                        \
        std::string _h = (haystack);                                        \
        std::string _n = (needle);                                          \
        if (_h.find(_n) != std::string::npos)                               \
        {                                                                   \
            recordPass();                                                   \
            std::cerr << "  [PASS] " << __FILE__ << ":" << __LINE__         \
                      << " | Contains: " << toStr(_n)                       \
                      << " | Actual: " << toStr(_h) << "\n";                \
        }                                                                   \
        else                                                                \
        {                                                                   \
            recordFail(__FILE__, __LINE__, #haystack " contains " #needle); \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__         \
                      << "\n    Expected output to contain: " << toStr(_n)  \
                      << "\n    Actual output:\n" << _h << std::endl;       \
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

// Helper to compute exact expected embedding output string for a token sequence
std::string computeExpectedEmbeddingOutput(Model &model, const std::vector<std::string> &tokens)
{
    std::size_t embSize = model.getEmbeddingSize();
    GPU refGpu(1, embSize);
    GenerativeRequest refReq(0, model, 0, "seed", "", 0);
    refGpu.reserve(1, refReq);
    std::size_t slot = refGpu.allocateBlock(refReq);
    refGpu.acquire(slot);

    std::vector<int> sums(embSize, 0);
    for (const std::string &tok : tokens)
    {
        model.generateTokenEmbedding({tok}, refGpu, slot, 0);
        for (std::size_t j = 0; j < embSize; ++j)
        {
            sums[j] += refGpu.readFromBlock(slot, j);
        }
    }
    refGpu.release(slot);

    std::ostringstream oss;
    oss << "[";
    for (std::size_t j = 0; j < embSize; ++j)
    {
        if (j > 0)
            oss << ", ";
        int val = tokens.empty() ? 0 : (sums[j] / static_cast<int>(tokens.size()));
        oss << val;
    }
    oss << "]";
    return oss.str();
}

// ============================================================================
// 1. MemoryBlock, KVBlock, EmbeddingWorkspace & Rule of 5 Tests (Bottom Layer)
// ============================================================================
void testMemoryBlocksRuleOf5()
{
    BEGIN_SECTION(kAllSectionNames[0]);

    SUBTEST("KVBlock zero-init, read/write & refCount");
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
    kv.release();                                  // refCount 2 -> 1
    ASSERT_EQ(kv.release(), IdleAction::KeepCached); // refCount 1 -> 0

    SUBTEST("KVBlock Rule of 5: copy constructor");
    kv.acquire();
    KVBlock kvCopy(kv);
    ASSERT_EQ(kvCopy.readFromBlock(0), 11);
    ASSERT_EQ(kvCopy.readFromBlock(2), 42);
    kvCopy.writeToBlock(2, 99);
    ASSERT_EQ(kv.readFromBlock(2), 42); // Original untouched

    SUBTEST("KVBlock Rule of 5: copy assignment & self-assignment");
    KVBlock kvAssigned(2);
    kvAssigned = kv;
    ASSERT_EQ(kvAssigned.readFromBlock(0), 11);
    ASSERT_EQ(kvAssigned.readFromBlock(2), 42);
    kvAssigned.writeToBlock(2, 77);
    ASSERT_EQ(kv.readFromBlock(2), 42);
    KVBlock &kvSelfRef = kvAssigned;
    kvAssigned = kvSelfRef;
    ASSERT_EQ(kvAssigned.readFromBlock(2), 77);

    SUBTEST("KVBlock Rule of 5: move constructor & move assignment");
    KVBlock kvMoved(std::move(kvCopy));
    ASSERT_EQ(kvMoved.readFromBlock(2), 99);

    KVBlock kvMoveAssigned(2);
    kvMoveAssigned.writeToBlock(0, -5);
    kvMoveAssigned = std::move(kvMoved);
    ASSERT_EQ(kvMoveAssigned.readFromBlock(2), 99);

    SUBTEST("KVBlock polymorphic clone() & virtual destructor");
    MemoryBlock *kvBasePtr = &kv;
    MemoryBlock *clonedKV = kvBasePtr->clone();
    ASSERT_EQ(clonedKV->readFromBlock(2), 42);
    ASSERT_EQ(clonedKV->onLastRelease(), IdleAction::KeepCached);
    clonedKV->writeToBlock(2, -1);
    ASSERT_EQ(kv.readFromBlock(2), 42);
    delete clonedKV;

    kv.release();

    SUBTEST("EmbeddingWorkspace zero-init, read/write & refCount");
    EmbeddingWorkspace ew(4);
    for (std::size_t i = 0; i < 4; ++i)
    {
        ASSERT_EQ(ew.readFromBlock(i), 0);
    }
    ew.writeToBlock(1, 55);
    ew.writeToBlock(3, -12);
    ew.acquire();
    ASSERT_EQ(ew.release(), IdleAction::Destroy);

    SUBTEST("EmbeddingWorkspace Rule of 5: copy constructor");
    ew.acquire();
    EmbeddingWorkspace ewCopy(ew);
    ASSERT_EQ(ewCopy.readFromBlock(1), 55);
    ASSERT_EQ(ewCopy.readFromBlock(3), -12);
    ewCopy.writeToBlock(1, 88);
    ASSERT_EQ(ew.readFromBlock(1), 55);

    SUBTEST("EmbeddingWorkspace Rule of 5: copy assignment & self-assignment");
    EmbeddingWorkspace ewAssigned(2);
    ewAssigned = ew;
    ASSERT_EQ(ewAssigned.readFromBlock(1), 55);
    ASSERT_EQ(ewAssigned.readFromBlock(3), -12);
    ewAssigned.writeToBlock(1, 66);
    ASSERT_EQ(ew.readFromBlock(1), 55);
    EmbeddingWorkspace &ewSelfRef = ewAssigned;
    ewAssigned = ewSelfRef;
    ASSERT_EQ(ewAssigned.readFromBlock(1), 66);

    SUBTEST("EmbeddingWorkspace Rule of 5: move constructor & move assignment");
    EmbeddingWorkspace ewMoved(std::move(ewCopy));
    ASSERT_EQ(ewMoved.readFromBlock(1), 88);

    EmbeddingWorkspace ewMoveAssigned(2);
    ewMoveAssigned.writeToBlock(0, 123);
    ewMoveAssigned = std::move(ewMoved);
    ASSERT_EQ(ewMoveAssigned.readFromBlock(1), 88);

    SUBTEST("EmbeddingWorkspace polymorphic clone() & virtual destructor");
    MemoryBlock *ewBasePtr = &ew;
    MemoryBlock *clonedEW = ewBasePtr->clone();
    ASSERT_EQ(clonedEW->readFromBlock(1), 55);
    ASSERT_EQ(clonedEW->onLastRelease(), IdleAction::Destroy);
    clonedEW->writeToBlock(1, -9);
    ASSERT_EQ(ew.readFromBlock(1), 55);
    delete clonedEW;

    ew.release();
    END_SECTION();
}

// ============================================================================
// 2. Model Tests
// ============================================================================
void testModel()
{
    BEGIN_SECTION(kAllSectionNames[1]);

    Model m("llama", 4);

    SUBTEST("Model::tokenizer alphanumeric splitting");
    std::vector<std::string> tokens = m.tokenizer("Hello, world! 123  a-b_c?");
    std::vector<std::string> expected = {"Hello", "world", "123", "abc"};
    ASSERT_EQ(tokens, expected);

    std::vector<std::string> onlyAlnum = m.tokenizer("the cat sat");
    ASSERT_EQ(onlyAlnum.size(), static_cast<std::size_t>(3));
    if (onlyAlnum.size() >= 3)
    {
        ASSERT_EQ(onlyAlnum[0], "the");
        ASSERT_EQ(onlyAlnum[2], "sat");
    }

    SUBTEST("Model::generateTokenEmbedding determinism");
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
    END_SECTION();
}

// ============================================================================
// 3. GPU, Prefix Caching, Eviction, Cross-GPU Copy & Rule of 5 Tests
// ============================================================================
void testGPU()
{
    BEGIN_SECTION(kAllSectionNames[2]);

    Model llama("llama", 2);
    Model otherModel("mistral", 2);
    // blockSize = 4, embeddingSize = 2 => t = 2 tokens per block
    GPU gpu(3, 4);

    SUBTEST("GPU initial slots & getBlocksForMemory");
    ASSERT_EQ(gpu.availableSlots(), static_cast<std::size_t>(3));
    ASSERT_EQ(gpu.getBlocksForMemory(0), static_cast<std::size_t>(0));
    ASSERT_EQ(gpu.getBlocksForMemory(4), static_cast<std::size_t>(1));
    ASSERT_EQ(gpu.getBlocksForMemory(5), static_cast<std::size_t>(2));

    GenerativeRequest r1(1, llama, 0, "a b c d", "", 0);
    GenerativeRequest r2(2, llama, 0, "a b e f", "", 0);

    SUBTEST("GPU reserve & unreserve");
    gpu.reserve(2, r1);
    ASSERT_EQ(gpu.availableSlots(), static_cast<std::size_t>(1));
    gpu.unreserve(1, r1);
    ASSERT_EQ(gpu.availableSlots(), static_cast<std::size_t>(2));
    gpu.reserve(1, r1); // back to 2 reserved

    SUBTEST("GPU block allocation & prefix caching");
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

    ASSERT_EQ(gpu.availableSlots(), static_cast<std::size_t>(1));

    SUBTEST("GPU::findLongestCachedPrefix");
    PrefixMatch pm = gpu.findLongestCachedPrefix("llama", 2, {"a", "b", "c", "d", "x"});
    ASSERT_EQ(pm.matchedTokens, static_cast<std::size_t>(4));
    ASSERT_EQ(pm.blockIds.size(), static_cast<std::size_t>(2));
    if (pm.blockIds.size() >= 2)
    {
        ASSERT_EQ(pm.blockIds[0], s0);
        ASSERT_EQ(pm.blockIds[1], s1);
    }

    PrefixMatch pmWrongModel = gpu.findLongestCachedPrefix("mistral", 2, {"a", "b", "c", "d"});
    ASSERT_EQ(pmWrongModel.matchedTokens, static_cast<std::size_t>(0));
    ASSERT_TRUE(pmWrongModel.blockIds.empty());

    SUBTEST("GPU duplicate prefix key handling");
    gpu.reserve(1, r2);
    std::size_t s2 = gpu.allocateBlock(r2);
    ASSERT_EQ(s2, static_cast<std::size_t>(2));
    gpu.acquire(s2);
    gpu.assignPrefixToBlock(s2, {"a", "b"}, "llama"); // duplicate!
    gpu.release(s2);                                  // no prefix key => deleted immediately
    ASSERT_EQ(gpu.availableSlots(), static_cast<std::size_t>(1));

    SUBTEST("GPU release & idle caching");
    gpu.release(s0);
    gpu.release(s1);
    ASSERT_EQ(gpu.availableSlots(), static_cast<std::size_t>(3));

    SUBTEST("GPU::copyMemoryFromOtherGPU");
    GPU gpu2(3, 4);
    gpu2.copyMemoryFromOtherGPU(gpu, {s0, s1});
    PrefixMatch pm2 = gpu2.findLongestCachedPrefix("llama", 2, {"a", "b", "c", "d"});
    ASSERT_EQ(pm2.matchedTokens, static_cast<std::size_t>(4));
    if (pm2.blockIds.size() >= 2)
    {
        ASSERT_EQ(gpu2.readFromBlock(pm2.blockIds[0], 0), 111);
        ASSERT_EQ(gpu2.readFromBlock(pm2.blockIds[1], 0), 222);
    }

    gpu2.copyMemoryFromOtherGPU(gpu, {s0, s1});
    ASSERT_EQ(gpu2.availableSlots(), static_cast<std::size_t>(3));

    SUBTEST("GPU eviction order");
    EmbeddingRequest e1(3, llama, 0, "hi");
    gpu.reserve(2, e1);
    std::size_t es0 = gpu.allocateBlock(e1);
    ASSERT_EQ(es0, static_cast<std::size_t>(2)); // empty slot first!
    gpu.acquire(es0);
    std::size_t es1 = gpu.allocateBlock(e1);
    ASSERT_EQ(es1, static_cast<std::size_t>(0)); // lowest idle slot evicted!
    gpu.acquire(es1);

    PrefixMatch pmBroken = gpu.findLongestCachedPrefix("llama", 2, {"a", "b", "c", "d"});
    ASSERT_EQ(pmBroken.matchedTokens, static_cast<std::size_t>(0));

    SUBTEST("GPU Rule of 5: copy constructor");
    GPU gpuCopy(gpu);
    ASSERT_EQ(gpuCopy.availableSlots(), gpu.availableSlots());

    SUBTEST("GPU Rule of 5: copy assignment & self-assignment");
    GPU gpuAssigned(1, 2);
    gpuAssigned = gpu;
    ASSERT_EQ(gpuAssigned.availableSlots(), gpu.availableSlots());
    GPU &gpuSelfRef = gpuAssigned;
    gpuAssigned = gpuSelfRef;
    ASSERT_EQ(gpuAssigned.availableSlots(), gpu.availableSlots());

    gpuAssigned.writeToBlock(es0, 0, 999);
    ASSERT_EQ(gpu.readFromBlock(es0, 0), 0);

    gpu.release(es0);
    gpu.release(es1);
    ASSERT_EQ(gpuCopy.availableSlots(), static_cast<std::size_t>(1));
    ASSERT_EQ(gpuAssigned.availableSlots(), static_cast<std::size_t>(1));

    SUBTEST("GPU Rule of 5: move constructor & move assignment");
    GPU gpuMoved(std::move(gpuCopy));
    ASSERT_EQ(gpuMoved.availableSlots(), static_cast<std::size_t>(1));

    GPU gpuMoveAssigned(2, 4);
    gpuMoveAssigned.copyMemoryFromOtherGPU(gpu2, {s0, s1});
    gpuMoveAssigned = std::move(gpuAssigned);
    ASSERT_EQ(gpuMoveAssigned.availableSlots(), static_cast<std::size_t>(1));
    ASSERT_EQ(gpuMoveAssigned.readFromBlock(es0, 0), 999);
    gpuMoveAssigned.release(es0);
    gpuMoveAssigned.release(es1);
    ASSERT_EQ(gpuMoveAssigned.availableSlots(), static_cast<std::size_t>(3));

    END_SECTION();
}

// ============================================================================
// 4. Request (GenerativeRequest & EmbeddingRequest) & Individual Output Tests
// ============================================================================
void testRequests()
{
    BEGIN_SECTION(kAllSectionNames[3]);

    Model llama("llama", 2);
    Model e5("e5", 2);
    // blockSize = 4, embeddingSize = 2 => t = 2 tokens/block
    GPU gpu(4, 4);

    SUBTEST("GenerativeRequest lifecycle & full output");
    GenerativeRequest g1(10, llama, 0, "a b c", "x y", 4);
    ASSERT_EQ(g1.getId(), 10);
    ASSERT_EQ(g1.getAdmissionCycle(), 0);
    ASSERT_EQ(g1.getMaxNeededMemory(), static_cast<std::size_t>(14));
    ASSERT_EQ(g1.getNeededMemory(gpu), static_cast<std::size_t>(14));
    ASSERT_EQ(g1.getCacheableTokens().size(), static_cast<std::size_t>(3));

    g1.reserveMemory(gpu);
    ASSERT_EQ(gpu.availableSlots(), static_cast<std::size_t>(0)); // 4 blocks reserved

    int acts = 0;
    while (!g1.finished())
    {
        g1.act(gpu);
        ++acts;
    }
    ASSERT_EQ(acts, 5);
    ASSERT_EQ(g1.getOutput(), "x y");
    ASSERT_EQ(g1.getNeedToUnreserve(), static_cast<std::size_t>(1));

    g1.deallocateMemory(gpu);
    ASSERT_EQ(gpu.availableSlots(), static_cast<std::size_t>(4));

    SUBTEST("GenerativeRequest prefix reuse & clone");
    PrefixMatch pm = gpu.findLongestCachedPrefix("llama", 2, {"a", "b", "c", "x", "z"});
    ASSERT_EQ(pm.matchedTokens, static_cast<std::size_t>(4));

    GenerativeRequest g2(11, llama, 1, "a b c x z", "w", 1);
    ASSERT_EQ(g2.getNeededMemory(gpu), static_cast<std::size_t>(4));

    Model llamaCopy("llama", 2);
    Request *g2Clone = g2.clone(llamaCopy);
    ASSERT_EQ(g2Clone->getId(), 11);
    ASSERT_TRUE(&g2Clone->getModel() == &llamaCopy);
    delete g2Clone;

    SUBTEST("GenerativeRequest truncated, zero-max, and empty scripted outputs");
    GenerativeRequest gTrunc(12, llama, 0, "one two", "alpha beta gamma delta", 2);
    gTrunc.reserveMemory(gpu);
    int truncActs = 0;
    while (!gTrunc.finished())
    {
        gTrunc.act(gpu);
        ++truncActs;
    }
    ASSERT_EQ(truncActs, 4);
    ASSERT_EQ(gTrunc.getOutput(), "alpha beta");
    gTrunc.deallocateMemory(gpu);

    GenerativeRequest gZeroOut(13, llama, 0, "only prompt", "ignored tokens", 0);
    gZeroOut.reserveMemory(gpu);
    int zeroActs = 0;
    while (!gZeroOut.finished())
    {
        gZeroOut.act(gpu);
        ++zeroActs;
    }
    ASSERT_EQ(zeroActs, 2);
    ASSERT_EQ(gZeroOut.getOutput(), "");
    gZeroOut.deallocateMemory(gpu);

    GenerativeRequest gEmptyScripted(14, llama, 0, "hello", "", 3);
    gEmptyScripted.reserveMemory(gpu);
    int emptyScriptedActs = 0;
    while (!gEmptyScripted.finished())
    {
        gEmptyScripted.act(gpu);
        ++emptyScriptedActs;
    }
    ASSERT_EQ(emptyScriptedActs, 1);
    ASSERT_EQ(gEmptyScripted.getOutput(), "");
    gEmptyScripted.deallocateMemory(gpu);

    SUBTEST("EmbeddingRequest workspace, mean output & clone");
    EmbeddingRequest emb(20, e5, 0, "hello world test");
    ASSERT_EQ(emb.getMaxNeededMemory(), static_cast<std::size_t>(4));
    ASSERT_EQ(emb.getNeededMemory(gpu), static_cast<std::size_t>(4));
    ASSERT_TRUE(emb.getCacheableTokens().empty());

    Model e5Copy("e5", 2);
    Request *embClone = emb.clone(e5Copy);
    ASSERT_EQ(embClone->getId(), 20);
    ASSERT_TRUE(&embClone->getModel() == &e5Copy);
    delete embClone;

    emb.reserveMemory(gpu);
    ASSERT_EQ(emb.getNeedToUnreserve(), static_cast<std::size_t>(0));

    int embActs = 0;
    while (!emb.finished())
    {
        emb.act(gpu);
        ++embActs;
    }
    ASSERT_EQ(embActs, 3);
    std::string expectedEmbOut = computeExpectedEmbeddingOutput(e5, {"hello", "world", "test"});
    ASSERT_EQ(emb.getOutput(), expectedEmbOut);

    emb.deallocateMemory(gpu);
    ASSERT_EQ(gpu.availableSlots(), static_cast<std::size_t>(4));
    ASSERT_EQ(emb.getOutput(), expectedEmbOut);

    SUBTEST("EmbeddingRequest single-token output");
    EmbeddingRequest embSingle(21, e5, 0, "solo");
    embSingle.reserveMemory(gpu);
    while (!embSingle.finished())
    {
        embSingle.act(gpu);
    }
    ASSERT_EQ(embSingle.getOutput(), computeExpectedEmbeddingOutput(e5, {"solo"}));
    embSingle.deallocateMemory(gpu);
    ASSERT_EQ(gpu.availableSlots(), static_cast<std::size_t>(4));

    END_SECTION();
}

// ============================================================================
// 5. SchedulingPolicy Tests (Exact Example from Assignment Section 4.4.6)
// ============================================================================
void testSchedulingPolicies()
{
    BEGIN_SECTION(kAllSectionNames[4]);

    Model llama("llama", 4);
    Model e5("e5", 4);
    GPU gpu(10, 4);

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

    SUBTEST("ArrivalOrderPolicy, ShortestJobPolicy & LongestPrefixPolicy");
    ArrivalOrderPolicy arrival;
    ShortestJobPolicy shortest;
    LongestPrefixPolicy longest;

    std::vector<int> ordArrival = arrival.choose(eligible, gpus);
    std::vector<int> expArrival = {0, 1, 2, 3};
    ASSERT_EQ(ordArrival, expArrival);

    std::vector<int> ordShortest = shortest.choose(eligible, gpus);
    std::vector<int> expShortest = {1, 2, 0, 3};
    ASSERT_EQ(ordShortest, expShortest);

    std::vector<int> ordLongest = longest.choose(eligible, gpus);
    std::vector<int> expLongest = {3, 2, 0, 1};
    ASSERT_EQ(ordLongest, expLongest);

    SUBTEST("SchedulingPolicy polymorphic clone()");
    SchedulingPolicy *clonedArrival = arrival.clone();
    ASSERT_EQ(clonedArrival->choose(eligible, gpus), expArrival);
    delete clonedArrival;

    SchedulingPolicy *clonedShortest = shortest.clone();
    ASSERT_EQ(clonedShortest->choose(eligible, gpus), expShortest);
    delete clonedShortest;

    SchedulingPolicy *clonedLongest = longest.clone();
    ASSERT_EQ(clonedLongest->choose(eligible, gpus), expLongest);
    delete clonedLongest;

    END_SECTION();
}

// ============================================================================
// 6. Server Overall End-to-End Output Tests
// ============================================================================
void testServerOverallOutput()
{
    BEGIN_SECTION(kAllSectionNames[5]);

    SUBTEST("Server multi-request end-to-end transcript");
    const std::string cfgPath = "/tmp/spl_overall_output_config.json";
    {
        std::ofstream out(cfgPath);
        out << R"({
            "scheduling_policy": "longestPrefix",
            "block_size": 4,
            "gpus": [
                { "memory_bytes": 16 },
                { "memory_bytes": 16 }
            ],
            "models": [
                { "id": "gpt", "embedding_size": 4 },
                { "id": "embedder", "embedding_size": 4 }
            ],
            "requests": [
                {
                    "id": 10,
                    "type": "generative",
                    "model": "gpt",
                    "cycle": 0,
                    "prompt": "a b",
                    "scripted_output": "c d e",
                    "max_output_tokens": 2
                },
                {
                    "id": 20,
                    "type": "generative",
                    "model": "gpt",
                    "cycle": 0,
                    "prompt": "too big",
                    "scripted_output": "x",
                    "max_output_tokens": 10
                },
                {
                    "id": 30,
                    "type": "text_embedder",
                    "model": "embedder",
                    "cycle": 1,
                    "prompt": "x y"
                },
                {
                    "id": 40,
                    "type": "generative",
                    "model": "gpt",
                    "cycle": 4,
                    "prompt": "a b x",
                    "scripted_output": "z",
                    "max_output_tokens": 1
                }
            ]
        })";
    }

    Model refEmbedder("embedder", 4);
    std::string expectedReq30Out = computeExpectedEmbeddingOutput(refEmbedder, {"x", "y"});

    Server server(cfgPath);
    std::string actualOutput;
    {
        CoutCapture cap;
        server.serve();
        actualOutput = cap.str();
    }

    std::string expectedOutput =
        "rejected 20 cycle=0\n"
        "finished 30 cycle=2 gpu=1: " + expectedReq30Out + "\n"
        "finished 10 cycle=3 gpu=0: c d\n"
        "finished 40 cycle=5 gpu=0: z\n"
        "summary: cycles=6 cache_hits=2 evictions=2\n";

    ASSERT_EQ(actualOutput, expectedOutput);
    std::remove(cfgPath.c_str());

    SUBTEST("Server empty requests list transcript");
    const std::string emptyCfgPath = "/tmp/spl_empty_output_config.json";
    {
        std::ofstream out(emptyCfgPath);
        out << R"({
            "scheduling_policy": "shortestJob",
            "block_size": 8,
            "gpus": [
                { "memory_bytes": 32 }
            ],
            "models": [
                { "id": "gpt", "embedding_size": 4 }
            ],
            "requests": []
        })";
    }

    Server emptyServer(emptyCfgPath);
    std::string emptyActualOutput;
    {
        CoutCapture cap;
        emptyServer.serve();
        emptyActualOutput = cap.str();
    }
    ASSERT_EQ(emptyActualOutput, "summary: cycles=0 cache_hits=0 evictions=0\n");
    std::remove(emptyCfgPath.c_str());

    END_SECTION();
}

// ============================================================================
// 7. Server Mid-Run Rule of 5 Tests
// ============================================================================
void testServerAndRuleOf5()
{
    BEGIN_SECTION(kAllSectionNames[6]);

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

    SUBTEST("Server mid-run copy constructor, copy assignment & self-assignment");
    Server original(cfgPath);

    std::string prefixOut;
    {
        CoutCapture cap;
        ASSERT_TRUE(original.step()); // cycle 0 (request 5 rejected, request 1 admitted)
        ASSERT_TRUE(original.step()); // cycle 1 (request 2 admitted)
        prefixOut = cap.str();
    }
    ASSERT_CONTAINS(prefixOut, "rejected 5 cycle=0");

    Server copied(original);

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

    ASSERT_EQ(origRest, copyRest);
    ASSERT_EQ(origRest, copyAssignedRest);
    ASSERT_CONTAINS(origRest, "finished 1");
    ASSERT_CONTAINS(origRest, "paris is the");
    ASSERT_CONTAINS(origRest, "finished 2");
    ASSERT_CONTAINS(origRest, "finished 3");
    ASSERT_CONTAINS(origRest, "hi there");
    ASSERT_CONTAINS(origRest, "finished 4");
    ASSERT_CONTAINS(origRest, "rome");
    ASSERT_CONTAINS(origRest, "summary:");

    SUBTEST("Server mid-run move constructor & move assignment");
    Server src(cfgPath);
    {
        CoutCapture cap;
        ASSERT_TRUE(src.step());
    }
    Server dst(std::move(src));

    std::string movedFromOut;
    {
        CoutCapture cap;
        src.serve();
        movedFromOut = cap.str();
    }
    ASSERT_EQ(movedFromOut, "summary: cycles=0 cache_hits=0 evictions=0\n");

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
    END_SECTION();
}

int main()
{
    installCrashHandlers();

    runSectionGuarded(testMemoryBlocksRuleOf5);
    runSectionGuarded(testModel);
    runSectionGuarded(testGPU);
    runSectionGuarded(testRequests);
    runSectionGuarded(testSchedulingPolicies);
    runSectionGuarded(testServerOverallOutput);
    runSectionGuarded(testServerAndRuleOf5);

    printTestSummary();

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