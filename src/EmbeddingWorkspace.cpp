#include "MemoryBlock.h"

EmbeddingWorkspace::EmbeddingWorkspace(std::size_t size) : MemoryBlock(size) {}

EmbeddingWorkspace::EmbeddingWorkspace(const EmbeddingWorkspace &other)
    : MemoryBlock(other) {}

IdleAction EmbeddingWorkspace::onLastRelease() const
{
    return IdleAction::Destroy;
}

EmbeddingWorkspace::~EmbeddingWorkspace() {}

MemoryBlock *EmbeddingWorkspace::clone() const
{
    return new EmbeddingWorkspace(*this);
}