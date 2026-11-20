#include "Request.h"
#include "Server.h"
#include "Model.h" 

Request ::Request(int id, const Model& model, int admissionCycle, std::string prompt)
    : id(id), model(model), admissionCycle(admissionCycle), tokens(model.tokenizer(prompt)),
     reservedBlocks(0)
{
}
GenerativeRequest::GenerativeRequest(int id, const Model& model, int admissionCycle, std::string prompt, std::string scriptedOutput,
    std::size_t maxOutputTokens)
    : Request(id, model, admissionCycle, prompt), scriptedOutput(scriptedOutput), maxOutputTokens(maxOutputTokens)
{
}
EmbeddingRequest::EmbeddingRequest(int id, const Model& model, int admissionCycle, std::string prompt)
    : Request(id, model, admissionCycle, prompt)
{
}
const Model& Request::getModel() const
{
    return model;
}
 int Request::getId() const
{
    return id;
}
int Request::getAdmissionCycle() const
{
    return admissionCycle;
}
bool EmbeddingRequest::finished() const
{
    return true;
}
bool GenerativeRequest::finished() const
{
    return tokens.empty() ||maxOutputTokens == 0 ;
}
void GenerativeRequest::act(GPU& gpu)
{
    if(!finished())
    {
        model.generateTokenEmbedding(tokens, gpu, memoryBlocks[0], memoryBlocks[1]);
    }
    
    
}