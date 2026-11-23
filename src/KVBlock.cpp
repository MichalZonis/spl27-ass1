#include "MemoryBlock.h"

KVBlock::KVBlock(std::size_t size) : MemoryBlock(size) {}

KVBlock::KVBlock(const KVBlock &other) : MemoryBlock(other) {}

KVBlock::KVBlock(KVBlock &&other) : MemoryBlock(other) {}

IdleAction KVBlock::onLastRelease() const
{
    return IdleAction::KeepCached;
}

MemoryBlock *KVBlock::clone() const
{
    return new KVBlock(*this);
}

KVBlock &KVBlock::operator=(const KVBlock &other)
{
    
}

KVBlock::~KVBlock() {}