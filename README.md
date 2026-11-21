### 1. Request Lifecycle & Server Flow (`Server::step`)

```mermaid
flowchart TD
    Start(["Start of New Cycle"]) --> Arrival["Collect requests where admissionCycle == currentCycle"]
    
    Arrival --> RejectCheck{"MaxNeededBlocks > blocksNum<br/>on ALL GPUs?"}
    RejectCheck -- Yes --> Rejected["Print 'rejected' & delete request"]
    RejectCheck -- No --> Waiting["Move to waitingRequests"]
    
    Waiting --> Policy["SchedulingPolicy::choose<br/>Order waiting requests by priority"]
    Policy --> AdmitLoop["Iterate through ordered waiting requests"]
    
    AdmitLoop --> FitCheck{"Any GPU with enough availableSlots?<br/>newBlocks + idlePrefixBlocks <= availableSlots"}
    FitCheck -- No --> StayWait["Remains in waitingRequests<br/>Continue checking next request"]
    
    FitCheck -- Yes --> ChooseGPU["Select best GPU:<br/>1. Most cached prefix blocks<br/>2. Fewest blocks needed<br/>3. Lower GPU index"]
    
    ChooseGPU --> CrossGPU{"Does another GPU hold a<br/>longer cached prefix?"}
    CrossGPU -- Yes --> CopyPrefix["copyMemoryFromOtherGPU<br/>Copy prefix blocks to chosen GPU"]
    CrossGPU -- No --> Reserve
    CopyPrefix --> Reserve["request.reserveMemory<br/>Acquire cached prefix as cache hits + reserve new blocks"]
    
    Reserve --> Active["Move to activeRequests"]
    
    Active --> ActLoop["Iterate activeRequests in activation order<br/>Call request.act()"]
    
    ActLoop --> Work["Perform 1 unit of work:<br/>Generative: compute KV for 1 token<br/>Embedding: add 1 token vector to running sum"]
    
    Work --> DoneCheck{"request.finished()?"}
    DoneCheck -- No --> NextCycle(["Next Cycle"])
    DoneCheck -- Yes --> Finish["Print 'finished'<br/>deallocateMemory: release blocks & unreserve unused<br/>Remove from activeRequests"]
    Finish --> NextCycle
```

---

### 2. Prefix Caching Mechanism (Lookup, Filling & Release)

```mermaid
flowchart TD
    subgraph Lookup ["1. Prefix Lookup (GPU::findLongestCachedPrefix)"]
        L1["Receive promptTokens & modelId"] --> L2["Divide into full blocks of t = blockSize / embeddingSize tokens"]
        L2 --> L3{"Cached block exists on GPU for<br/>modelId + first k*t tokens?"}
        L3 -- Yes --> L4["Add slot to match list<br/>Check next block k+1"]
        L4 --> L3
        L3 -- No --> L5["Stop at first missing block<br/>Return matched slots & h covered tokens"]
    end

    subgraph Fill ["2. Filling Blocks during act()"]
        F1["Write token KV vector to block<br/>Block has no key while filling"] --> F2{"Is block full?<br/>t tokens written"}
        F2 -- No --> F3["Block remains without a key"]
        F2 -- Yes --> F4["Call GPU::assignPrefixToBlock"]
        F4 --> F5{"Another block already has<br/>same modelId & prefixKey?"}
        F5 -- Yes --> F6["Not cached<br/>Remains without a key"]
        F5 -- No --> F7["Store modelId & prefixKey for this slot"]
    end

    subgraph Release ["3. Releasing a Block (GPU::release)"]
        R1["refCount--"] --> R2{"refCount == 0?"}
        R2 -- No --> R3["Block stays Active<br/>Still held by other requests"]
        R2 -- Yes --> R4{"Check onLastRelease()"}
        R4 -->|"EmbeddingWorkspace<br/>(Destroy)"| R5["Delete block<br/>Slot becomes nullptr"]
        R4 -->|"KVBlock<br/>(KeepCached)"| R6{"Does block have<br/>a prefixKey?"}
        R6 -->|"No (partial or duplicate)"| R7["Delete block & empty slot<br/>Not counted as eviction"]
        R6 -- Yes --> R8["Block becomes Idle<br/>Available for reuse or eviction"]
    end
```

### 3. General Program Workflow & Components

```mermaid
flowchart TD
    subgraph Init ["1. Initialization (Main.cpp & Config)"]
        Config[("config.json<br/>GPUs, Models, Policy, Requests")] --> Main["Main.cpp<br/>Server server(argv[1])"]
        Main --> ReadCfg["Server::readConfig()<br/>(via json.hpp)"]
        ReadCfg --> ServerInit["Server initializes & owns:<br/>• vector&lt;GPU&gt;<br/>• vector&lt;Model&gt;<br/>• SchedulingPolicy*<br/>• Pending Request*s"]
    end

    ServerInit --> ServeLoop["Server::serve()<br/>Loop while step() returns true"]

    subgraph Cycle ["2. Simulation Cycle (Server::step)"]
        Arrival["Arrival Phase<br/>Move due requests to waitingRequests<br/>(Reject & delete if too large for all GPUs)"]
        
        Arrival --> PolicyComp["SchedulingPolicy::choose()<br/>(ArrivalOrder / ShortestJob / LongestPrefix)<br/>Reads GPUs & waitingRequests (const)"]
        
        PolicyComp --> Admit["Admission Phase<br/>1. Pick best GPU (cached prefix, needed blocks, index)<br/>2. Copy longer prefix from other GPU if found<br/>3. request.reserveMemory(gpu) & move to activeRequests"]
        
        Admit --> Exec["Execution Phase<br/>For each active request: request.act(gpu)"]
        
        Exec --> Cleanup["Completion Phase<br/>If request.finished():<br/>Print 'finished', call deallocateMemory(gpu), remove"]
    end

    ServeLoop --> Arrival
    Cleanup -->|"Requests remain"| ServeLoop
    Cleanup -->|"All requests done"| Summary(["Server::serve() prints:<br/>summary: cycles, cache_hits, evictions"])

    subgraph ReqModel ["3. Requests & Models"]
        Req["Request (Abstract)<br/>• GenerativeRequest (prefill + decode)<br/>• EmbeddingRequest (token mean)<br/>Holds slot indices & non-owning const Model&"]
        Model["Model (Stateless)<br/>• tokenizer()<br/>• generateTokenEmbedding()"]
        Req -->|"Requests token vector"| Model
    end

    subgraph GPUMem ["4. GPU & Memory Management"]
        GPU["GPU (Owns all blocks)<br/>• Fixed slots: vector&lt;MemoryBlock*&gt;<br/>• Reservations, Prefix Map, Stats<br/>• allocateBlock(), acquire(), release()"]
        Blocks["MemoryBlock (Abstract)<br/>• KVBlock (shared, KeepCached)<br/>• EmbeddingWorkspace (private, Destroy)"]
        GPU -->|"Allocates, owns, evicts & deletes"| Blocks
    end

    Admit <-->|"findLongestCachedPrefix()<br/>copyMemoryFromOtherGPU()<br/>reserve() & acquire()"| GPU
    Exec <-->|"act()"| Req
    Req -->|"Double Dispatch:<br/>allocateOneBlock() -> addMemoryBlock()"| GPU
    Model -->|"writeToBlock()"| GPU
    Cleanup -->|"unreserve() & release()<br/>(triggers onLastRelease)"| GPU
```

### 4. Function Call Flow (Who Calls Whom & Where)

```mermaid
flowchart TD
    subgraph Main ["Main.cpp"]
        M_main["main()"]
    end

    subgraph Server ["Server"]
        S_ctor["Server(configPath)"]
        S_read["readConfig()"]
        S_serve["serve()"]
        S_step["step()"]
    end

    subgraph Policy ["SchedulingPolicy"]
        P_choose["choose(eligible, gpus)"]
    end

    subgraph Request ["Request (Generative / Embedding)"]
        R_ctor["Request()"]
        R_max["getMaxNeededMemory()"]
        R_needed["getNeededMemory(gpu)"]
        R_cache["getCacheableTokens()"]
        R_res["reserveMemory(gpu)"]
        R_act["act(gpu)"]
        R_fin["finished() & getOutput()"]
        R_dealloc["deallocateMemory(gpu)"]
        R_unres["getNeedToUnreserve()"]
        R_alloc1["allocateOneBlock(gpu, slot)"]
    end

    subgraph Model ["Model"]
        Mod_tok["tokenizer()"]
        Mod_emb["generateTokenEmbedding()"]
    end

    subgraph GPU ["GPU"]
        G_avail["availableSlots()"]
        G_blocks["getBlocksForMemory()"]
        G_find["findLongestCachedPrefix()"]
        G_copy["copyMemoryFromOtherGPU()"]
        G_res["reserve() / unreserve()"]
        G_alloc["allocateBlock(request)"]
        G_add["addMemoryBlock(req, slot)"]
        G_assign["assignPrefixToBlock()"]
        G_acq["acquire(slot)"]
        G_rel["release(slot)"]
        G_rw["readFromBlock() / writeToBlock()"]
    end

    subgraph MemoryBlock ["MemoryBlock (KVBlock / EmbeddingWorkspace)"]
        MB_acq["acquire()"]
        MB_rel["release() & onLastRelease()"]
        MB_rw["readFromBlock() / writeToBlock()"]
        MB_clone["clone()"]
    end

    %% Init
    M_main --> S_ctor
    S_ctor --> S_read
    S_ctor --> R_ctor
    R_ctor --> Mod_tok
    M_main --> S_serve
    S_serve --> S_step

    %% Step: Arrival & Scheduling
    S_step -->|"Rejection check"| R_max
    S_step -->|"Rejection check"| G_blocks
    S_step --> P_choose
    P_choose -->|"ShortestJob"| R_max
    P_choose -->|"ShortestJob"| G_blocks
    P_choose -->|"LongestPrefix"| G_find

    %% Step: Admission
    S_step -->|"Find best GPU"| G_find
    S_step -->|"Check fit"| G_avail
    S_step -->|"Check fit"| R_needed
    S_step -->|"Cross-GPU prefix"| G_copy
    G_copy --> MB_clone
    S_step -->|"Admit"| R_res

    %% Request::reserveMemory
    R_res -->|"Gen: find prefix"| G_find
    R_res -->|"Gen: prefix hits / Emb: workspace"| G_acq
    R_res -->|"Reserve needed"| G_res
    R_res -->|"Emb only: alloc workspace"| G_alloc

    %% Step: Execution
    S_step --> R_act
    R_act -->|"Gen: when new block needed"| G_alloc
    R_act -->|"Gen: after alloc"| G_acq
    R_act -->|"Compute token vector"| Mod_emb
    Mod_emb -->|"Write vector"| G_rw
    R_act -->|"Emb: update running sum"| G_rw
    R_act -->|"Gen: when block is full"| G_assign

    %% Double Dispatch
    G_alloc -->|"Double Dispatch (1)"| R_alloc1
    R_alloc1 -->|"Double Dispatch (2)"| G_add

    %% Step: Completion
    S_step --> R_fin
    S_step -->|"If finished"| R_dealloc
    R_dealloc --> R_unres
    R_dealloc --> G_res
    R_dealloc --> G_rel

    %% GPU -> MemoryBlock
    G_acq --> MB_acq
    G_rel --> MB_rel
    G_rw --> MB_rw
```
````[cite: 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
