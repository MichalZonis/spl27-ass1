#include "schedulingPolicy.h"

std::vector<int> ArrivalOrderPolicy::choose(const std::vector<const Request*>& eligible, 
    const std::vector<GPU>& gpus) const
{
    std::vector<int> order;
    auto copy = eligible.clone();
    int index = 0;
    for(int j=0; j<eligible.size(); ++j)
    {
    for (int i = 0; i < eligible.size(); ++i) 
    {
        if(copy[i]!=nullptr)
        if (eligible[i]->getAdmissionCycle() < eligible[index]->getAdmissionCycle()) 
        {
            index = i;
        }
    }
    order.push_back(index);
    copy[index] = nullptr; // Mark as chosen
}
    return order;
}

std::vector<int> ShortestJobPolicy::choose(const std::vector<const Request*>& eligible,
    const std:vector<GPU>& gpus) const
{
    std::vector<int> order;
    auto copy = eligible.clone();
    int index = 0;
    for(int j=0; j<eligible.size(); ++j)
    {
    for (int i = 0; i < eligible.size(); ++i)
    {
        if(copy[i]!=nullptr)
        if (eligible[i]->getMaxOutputTokens() < eligible[index]->getMaxOutputTokens()) 
        {   
            index = i;
        }
    }
    order.push_back(index); 
    copy[index] =nullptr;
}
return order;
}

  