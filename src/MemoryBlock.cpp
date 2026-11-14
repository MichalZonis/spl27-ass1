#include "MemoryBlock.h"

std::size_t MemoryBlock::getRefCount() const {
    return refCount;
}   