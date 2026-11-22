#include "MemoryBlock.h"

MemoryBlock::MemoryBlock(std::size_t size) : size(size), refCount(0) {
    data = new int[size];
}

MemoryBlock::MemoryBlock(const MemoryBlock &other) : size(other.size), refCount(0) {
    data = new int[size];
    std::copy(other.data, other.data + size, data);
}

MemoryBlock::~MemoryBlock(){
    delete[] data;
}

std::size_t MemoryBlock::getRefCount() const
{
    return refCount;
}

int MemoryBlock::readFromBlock(std::size_t offset) const
{
    return data[offset];
}

void MemoryBlock::writeToBlock(std::size_t offset, int value)
{
    data[offset] = value;
}

void MemoryBlock::acquire()
{
    refCount++;
}

IdleAction MemoryBlock::release()
{
    if (refCount > 0)
    {
        refCount--;
    }
    if (refCount == 0)
    {
        return onLastRelease();
    }
    return IdleAction::KeepCached;
}