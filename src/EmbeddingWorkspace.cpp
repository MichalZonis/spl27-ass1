#include "MemoryBlock.h"

EmbeddingWorkspace::EmbeddingWorkspace(std::size_t size) : MemoryBlock(size) {}

IdleAction EmbeddingWorkspace::onLastRelease() const
{
    return IdleAction::Destroy;
}