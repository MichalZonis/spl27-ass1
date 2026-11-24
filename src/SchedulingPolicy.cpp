#include "SchedulingPolicy.h"

std::vector<int> ArrivalOrderPolicy::choose(const std::vector<const Request*>& eligible,
                                            const std::vector<GPU>& gpus) const
{
    (void)gpus;
    std::vector<int> order;
    order.reserve(eligible.size());
    std::vector<bool> chosen(eligible.size(), false);

    for (std::size_t step = 0; step < eligible.size(); ++step) {
        int bestIdx = -1;
        for (std::size_t i = 0; i < eligible.size(); ++i) {
            if (chosen[i] || eligible[i] == nullptr) {
                continue;
            }
            if (bestIdx == -1) {
                bestIdx = static_cast<int>(i);
                continue;
            }
            const Request* cur = eligible[i];
            const Request* best = eligible[bestIdx];
            if (cur->getAdmissionCycle() < best->getAdmissionCycle() ||
                (cur->getAdmissionCycle() == best->getAdmissionCycle() && cur->getId() < best->getId())) {
                bestIdx = static_cast<int>(i);
            }
        }
        if (bestIdx != -1) {
            chosen[bestIdx] = true;
            order.push_back(bestIdx);
        }
    }
    return order;
}

SchedulingPolicy* ArrivalOrderPolicy::clone() const
{
    return new ArrivalOrderPolicy(*this);
}

std::vector<int> ShortestJobPolicy::choose(const std::vector<const Request*>& eligible,
                                           const std::vector<GPU>& gpus) const
{
    std::vector<int> order;
    order.reserve(eligible.size());
    if (gpus.empty()) {
        return order;
    }

    std::vector<bool> chosen(eligible.size(), false);
    const GPU& refGpu = gpus[0];

    for (std::size_t step = 0; step < eligible.size(); ++step) {
        int bestIdx = -1;
        for (std::size_t i = 0; i < eligible.size(); ++i) {
            if (chosen[i] || eligible[i] == nullptr) {
                continue;
            }
            if (bestIdx == -1) {
                bestIdx = static_cast<int>(i);
                continue;
            }
            const Request* cur = eligible[i];
            const Request* best = eligible[bestIdx];
            std::size_t curBlocks = refGpu.getBlocksForMemory(cur->getMaxNeededMemory());
            std::size_t bestBlocks = refGpu.getBlocksForMemory(best->getMaxNeededMemory());
            if (curBlocks < bestBlocks ||
                (curBlocks == bestBlocks && cur->getId() < best->getId())) {
                bestIdx = static_cast<int>(i);
            }
        }
        if (bestIdx != -1) {
            chosen[bestIdx] = true;
            order.push_back(bestIdx);
        }
    }
    return order;
}

SchedulingPolicy* ShortestJobPolicy::clone() const
{
    return new ShortestJobPolicy(*this);
}
// LongestPrefixPolicy implementation work in progress need to understand longest prefix in which gpu and how to compare between gpus and requests
std::vector<int> LongestPrefixPolicy::choose(const std::vector<const Request*>& eligible,
                                             const std::vector<GPU>& gpus) const
{
    std::vector<int> order;
    order.reserve(eligible.size());

    if (gpus.empty() || eligible.empty()) {
        return order;
    }

    auto getPrefixLength = [&](const Request* req) -> std::size_t {

        std::size_t maxMatched = 0;
        const auto& model = req->getModel();

        for (std::size_t gpuIndex = 0; gpuIndex < gpus.size(); ++gpuIndex) {
            auto prefix = gpus[gpuIndex].findLongestCachedPrefix(
                model.getId(),
                model.getEmbeddingSize(),
                req->getCacheableTokens()
            );
            if (prefix.matchedTokens > maxMatched) {
                maxMatched = prefix.matchedTokens;
            }
        }
        return maxMatched;
    };

    std::vector<bool> chosen(eligible.size(), false);

    for (std::size_t step = 0; step < eligible.size(); ++step) {
        int bestIdx = -1;
        std::size_t bestPrefix = 0;
        int bestId = 0; 

        for (std::size_t i = 0; i < eligible.size(); ++i) {
            if (chosen[i] || eligible[i] == nullptr) {
                continue;
            }

            const Request* cur = eligible[i];
            std::size_t curPrefix = getPrefixLength(cur);
            auto curId = cur->getId();

            bool isBetter = false;
            if (bestIdx == -1) {
                isBetter = true;
            } else if (curPrefix != bestPrefix) {
                isBetter = curPrefix > bestPrefix;
            } else {
                isBetter = curId < bestId;
            }

            if (isBetter) {
                bestIdx = static_cast<int>(i);
                bestPrefix = curPrefix;
                bestId = curId;
            }
        }

        if (bestIdx == -1) {
            break;
        }

        chosen[static_cast<std::size_t>(bestIdx)] = true;
        order.push_back(bestIdx);
    }

    return order;
}

SchedulingPolicy* LongestPrefixPolicy::clone() const
{
    return new LongestPrefixPolicy(*this);
}
