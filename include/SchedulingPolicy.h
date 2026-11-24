#ifndef SCHEDULING_POLICY_H_
#define SCHEDULING_POLICY_H

#include <vector>
#include "Request.h"
#include "GPU.h"
#include <cstddef>
#include "Model.h"

class SchedulingPolicy
{
public:
    virtual ~SchedulingPolicy() = default;
    virtual std::vector<int> choose(const std::vector<const Request *> &eligible, const std::vector<GPU> &gpus) const = 0;
    virtual SchedulingPolicy *clone() const = 0;
};

class ArrivalOrderPolicy : public SchedulingPolicy
{
    public: //Todo: remove
    std::vector<int> choose(
        const std::vector<const Request *> &eligible,
        const std::vector<GPU> &gpus) const override;

    SchedulingPolicy *clone() const override;
};
class ShortestJobPolicy : public SchedulingPolicy
{
    public: //Todo: remove

    std::vector<int> choose(
        const std::vector<const Request *> &eligible,
        const std::vector<GPU> &gpus) const override;

    SchedulingPolicy *clone() const override;
};
class LongestPrefixPolicy : public SchedulingPolicy
{
    public: //Todo: remove

    std::vector<int> choose(
        const std::vector<const Request *> &eligible,
        const std::vector<GPU> &gpus) const override;

    SchedulingPolicy *clone() const override;
};

#endif