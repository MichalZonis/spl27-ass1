#include "MemoryBlock.h"

KVBlock::KVBlock(std::size_t size) : MemoryBlock(size) {}

IdleAction KVBlock::onLastRelease() const{
return IdleAction::KeepCached;
}

MemoryBlock* KVBlock::clone() const {
    
}