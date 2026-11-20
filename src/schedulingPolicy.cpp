#include "SchedulingPolicy.h"
#include "Request.h"
#include "GPU.h"
#include <cstddef>
#include <vector>

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

std::vector<int> LongestPrefixPolicy::choose(const std::vector<const Request*>& eligible,
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
            if (curBlocks > bestBlocks ||
                (curBlocks == bestBlocks && cur->getId() > best->getId())) {
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

SchedulingPolicy* LongestPrefixPolicy::clone() const
{
    return new LongestPrefixPolicy(*this);
}
