class Value {
    constructor(data, inputs = []) {
        this.data = data;
        this.grad = 0;
        this._inputs = inputs;
        this._backward = () => {}; // leaves do nothing
    }

    // Primitives

    add(other) {
        if (!(other instanceof Value)) other = new Value(other);
        const out = new Value(this.data + other.data, [this, other]);
        out._backward = () => {
            // d(a + b)/da = 1, d(a + b)/db = 1
            this.grad  += out.grad * 1.0;
            other.grad += out.grad * 1.0;
        };
        return out;
    }

    mul(other) {
        if (!(other instanceof Value)) other = new Value(other);
        const out = new Value(this.data * other.data, [this, other]);
        out._backward = () => {
            // d(a * b)/da = b, d(a * b)/db = a
            this.grad  += other.data * out.grad;
            other.grad += this.data  * out.grad;
        };
        return out;
    }

    pow(exponent) {
        // exponent is a plain number, not a Value
        const out = new Value(Math.pow(this.data, exponent), [this]);
        out._backward = () => {
            // d(a^k)/da = k * a^(k-1)
            this.grad += exponent * Math.pow(this.data, exponent - 1) * out.grad;
        };
        return out;
    }

    relu() {
        const out = new Value(this.data > 0 ? this.data : 0, [this]);
        out._backward = () => {
            // d/dx max(0, x) = 1 if x > 0 else 0
            this.grad += (this.data > 0 ? 1 : 0) * out.grad;
        };
        return out;
    }

    exp() {
        const out = new Value(Math.exp(this.data), [this]);
        out._backward = () => {
            // d/dx exp(x) = exp(x), which is exactly out.data
            this.grad += out.data * out.grad;
        };
        return out;
    }

    log() {
        const out = new Value(Math.log(this.data), [this]);
        out._backward = () => {
            // d/dx log(x) = 1/x
            this.grad += (1 / this.data) * out.grad;
        };
        return out;
    }

    // Derived

    neg()      { return this.mul(-1); }
    sub(other) { return this.add((other instanceof Value ? other : new Value(other)).neg()); }
    div(other) { return this.mul((other instanceof Value ? other : new Value(other)).pow(-1)); }
    sqrt()     { return this.pow(0.5); }

    // Graph Traversal

    backward() {
        // Topological sort: inputs come before the nodes built from them.
        const topo = [];
        const visited = new Set();

        const visit = (node) => {
            if (visited.has(node)) return;
            visited.add(node);
            for (let i = 0; i < node._inputs.length; ++i) {
                visit(node._inputs[i]);
            }
            topo.push(node);
        };
        visit(this);

        // Seed: derivative of output w.r.t. itself.
        this.grad = 1;

        // Walk in reverse: roots first, leaves last.
        // Each node pushes gradient onto its _inputs via the chain rule.
        for (let i = topo.length - 1; i >= 0; --i) {
            topo[i]._backward();
        }
    }

    zeroGrad() { // Reset gradients across the whole graph
        const visited = new Set();
        const visit = (node) => {
            if (visited.has(node)) return;
            visited.add(node);
            node.grad = 0;
            for (let i = 0; i < node._inputs.length; ++i) {
                visit(node._inputs[i]);
            }
        };
        visit(this);
    }
}

class Matrix {  // Row major matrix
    constructor(numRows, numColumns) {
        this.rows = numRows;
        this.columns = numColumns;
        this.values = new Array(this.rows * this.columns);
        // Default matrix is uninitialized, call an init function
    }

    initToZeroes() {
        for (let i = 0; i < this.rows; ++i) {
            for (let j = 0; j < this.columns; ++j) {
                this.values[i * this.columns + j] = new Value(0.0);
            }
        }
    }

    initToSmallRandom() {
        // Xavier-ish: U(-scale, +scale) with scale = 1/sqrt(fan_in).
        // For a weight matrix used as `x @ W`, fan_in is the number of rows.
        const scale = 1 / Math.sqrt(this.rows);
        for (let i = 0; i < this.rows; ++i) {
            for (let j = 0; j < this.columns; ++j) {
                const r = (Math.random() - 0.5) * 2 * scale; // uniform in [-scale, +scale)
                this.values[i * this.columns + j] = new Value(r);
            }
        }
    }

    initToOnes() {
        for (let i = 0; i < this.rows; ++i) {
            for (let j = 0; j < this.columns; ++j) {
                this.values[i * this.columns + j] = new Value(1.0);
            }
        }
    }
    
    get(index) { 
        return this.values[index];
    }

    set(index, value) {
        if (value instanceof Value) {
            this.values[index] = value;
        }
        else {
            this.values[index] = new Value(value);
        }
    }

    transposed() { // returns new Matrix, columns/rows swapped.  Cells are the SAME Value refs — no graph nodes added. DOES NOT ALLOCATE NEW VALUES
        let result = new Matrix(this.columns, this.rows);

        for (let row = 0; row < this.rows; ++row) {
            for (let col = 0; col < this.columns; ++col) {
                const thisIndex = row * this.columns + col;
                const resIndex = col * this.rows + row;

                result.values[resIndex] = this.values[thisIndex];
            }
        }
        return result;
    }

    add(other) { // elementwise ops (return new Matrix)
        const result = new Matrix(this.rows, this.columns);
        // not initializing result, Values are calculated below

        for (let row = 0; row < this.rows; ++row) {
            for (let col = 0; col < this.columns; ++col) {
                const index = row * this.columns + col;
                result.values[index] = this.values[index].add(other.values[index]);
            }
        }

        return result;

    } 

    scale(scalar) { // this[i,j].mul(scalar) — plain number ok, Value.mul wraps it
        const result = new Matrix(this.rows, this.columns);
        // not initializing result, Values are calculated below

        for (let row = 0; row < this.rows; ++row) {
            for (let col = 0; col < this.columns; ++col) {
                const index = row * this.columns + col;
                result.values[index] = this.values[index].mul(scalar);
            }
        }

        return result;
    }
   
    mul(other) {  // ---- matmul ----
        const result = new Matrix(this.rows, other.columns);

        for (let row = 0; row < this.rows; ++row) {
            for (let col = 0; col < other.columns; ++col) {
                // Dot product of row `row` of this with column `col` of other.
                // Seed with the k=0 term, then accumulate. This keeps every cell
                // as a real Value-graph node so backward() flows through matmul.
                const aFirst = this.values[row * this.columns + 0];
                const bFirst = other.values[0 * other.columns + col];
                let sum = aFirst.mul(bFirst);

                for (let k = 1; k < this.columns; ++k) {
                    const a = this.values[row * this.columns + k];
                    const b = other.values[k * other.columns + col];
                    sum = sum.add(a.mul(b));
                }

                result.values[row * other.columns + col] = sum;
            }
        }

        return result;
    }

    causalMasked() {  // copy cells where j <= i; cells where j > i become new Value(-Infinity). ONLY PLACE WE CREATE A FRESH VALUE OUT OF THIN AIR
        const result = new Matrix(this.rows, this.columns); 
        // not initializing result, Values are calculated below

        for (let row = 0; row < this.rows; ++row) {
            for (let col = 0; col < this.columns; ++col) {
                const index = row * this.columns + col;
                if (col > row) {
                    result.values[index] = new Value(-Infinity);
                }
                else {
                    result.values[index] = this.values[index];
                }
            }
        }

        return result;
    }

    softMaxedRows() {
        const result = new Matrix(this.rows, this.columns);

        for (let row = 0; row < this.rows; ++row) {
            const base = row * this.columns;

            let rowMax = this.values[base + 0];
            for (let col = 1; col < this.columns; ++col) {
                const v = this.values[base + col];
                if (v.data > rowMax.data) rowMax = v;
            }

            const exps = new Array(this.columns);
            exps[0] = this.values[base + 0].sub(rowMax).exp();
            let rowSum = exps[0];
            for (let col = 1; col < this.columns; ++col) {
                exps[col] = this.values[base + col].sub(rowMax).exp();
                rowSum = rowSum.add(exps[col]);
            }

            for (let col = 0; col < this.columns; ++col) {
                result.values[base + col] = exps[col].div(rowSum);
            }
        }

        return result;
    }

    parameters() { 
        return this.values;
    }          
}

class RMSNorm { // Normalizes matrix per row
    constructor(featureDimensions) {
        this.learnedGamma = new Matrix(1, featureDimensions);
        this.learnedGamma.initToOnes();
        this.featureDimensions = featureDimensions;
    }

    forward(matrix) {
        const RMS_EPSILON = 1e-5;
        const sequenceLength = matrix.rows;
        const featureDimensions = matrix.columns;

        if (featureDimensions !== this.featureDimensions) {
            throw new Error(`RMSNorm expected ${this.featureDimensions} features, got ${featureDimensions}`);
        }

        const result = new Matrix(sequenceLength, featureDimensions);
        // No init — every cell is filled with a freshly computed Value below.

        for (let row = 0; row < sequenceLength; ++row) {
            const base = row * featureDimensions;

            const first = matrix.get(base);
            let sumSq = first.mul(first);
            for (let col = 1; col < featureDimensions; ++col) {
                const v = matrix.get(base + col);
                sumSq = sumSq.add(v.mul(v));
            }

            const meanSq = sumSq.mul(1 / featureDimensions);
            const rms    = meanSq.add(RMS_EPSILON).sqrt();

            const invRms = rms.pow(-1);

            for (let col = 0; col < featureDimensions; ++col) {
                const index      = base + col;
                const normalized = matrix.get(index).mul(invRms);
                const gamma      = this.learnedGamma.get(col); // gamma is (1, F), so col indexes it directly
                result.set(index, normalized.mul(gamma));
            }
        }

        return result;
    }

    parameters() {
        return this.learnedGamma.parameters();
    }
}

class AttentionHeadSingle { 
    constructor(feature_dim, head_dim) {
        this.learnedQ = new Matrix(feature_dim, head_dim);
        this.learnedQ.initToSmallRandom();
        this.learnedK = new Matrix(feature_dim, head_dim);
        this.learnedK.initToSmallRandom();
        this.learnedV = new Matrix(feature_dim, head_dim);
        this.learnedV.initToSmallRandom();
        this.learnedO = new Matrix(head_dim, feature_dim);
        this.learnedO.initToSmallRandom();
        this.head_dim = head_dim;
    }

    forward(layer_norm_matrix) { // Shape: sequence_length, feature_dim
        const sequenceLength = layer_norm_matrix.rows;

        let Q = layer_norm_matrix.mul(this.learnedQ); // Shape: sequence_length, head_dim
        let K = layer_norm_matrix.mul(this.learnedK); // Shape: sequence_length, head_dim
        let V = layer_norm_matrix.mul(this.learnedV); // Shape: sequence_length, head_dim

        let scores = Q.mul(K.transposed()); // Shape: sequence_length, sequence_length
        let scale = 1.0 / Math.sqrt(this.head_dim); // scalar
        scores = scores.scale(scale); // Element wise scalar multiplication

        scores = scores.causalMasked(); // Upper right is now -infinity
        let probabilities = scores.softMaxedRows(); // Shape: sequence_length, sequence_length
        let mixed = probabilities.mul(V); // Shape: sequence_length, head_dim
        let out = mixed.mul(this.learnedO); // Shape: sequence_length, feature_dim

        return out; // Output shape same as input shape
    }

    parameters() {
        const params = [];
        const matrices = [this.learnedQ, this.learnedK, this.learnedV, this.learnedO];
        for (let i = 0; i < matrices.length; ++i) {
            const matParams = matrices[i].parameters();
            for (let j = 0; j < matParams.length; ++j) {
                params.push(matParams[j]);
            }
        }
        return params;
    }
}

class Perceptron {
    constructor(feature_dim_size, hidden_dim_size) { 
        this.learnedUp = new Matrix(feature_dim_size, hidden_dim_size);
        this.learnedUp.initToSmallRandom();
        this.learnedDown = new Matrix(hidden_dim_size, feature_dim_size);
        this.learnedDown.initToSmallRandom();
        this.hiddenDim = hidden_dim_size;
    }

    activation(value) { // ReLU
        /*if (!(value instanceof Value)) {
            throw new Error(`calling Perceptron.activation on a non value class`);
        }*/
        return value.relu();
    }

    forward(layer_norm_matrix) { // Shape: sequence_length, feature_dim
        const sequenceLength = layer_norm_matrix.rows;

        let hidden = layer_norm_matrix.mul(this.learnedUp); // Shape: sequence_length, hidden_dim

        for (let i = 0; i < sequenceLength; ++i) {
            for (let j = 0; j < this.hiddenDim; ++j) {
                const index = i * this.hiddenDim + j;
                hidden.set(index, this.activation(hidden.get(index)));
            }
        }

        let out = hidden.mul(this.learnedDown); // Shape: sequence_length, feature_dim;
        return out;
    }

    parameters() {
        const params = [];
        const matrices = [this.learnedUp, this.learnedDown];
        for (let i = 0; i < matrices.length; ++i) {
            const matParams = matrices[i].parameters();
            for (let j = 0; j < matParams.length; ++j) {
                params.push(matParams[j]);
            }
        }
        return params;
    }
}

class TransformerBlock {
    constructor(feature_dim) {
        // By hard convention, head dim is usually feature dim divided by num heads
        const head_dim = feature_dim; // Feature Dim / Num Heads (one for now)

        // Again, hard convention is to make the hidden dimension 4x the feature dim
        const hidden_dim = feature_dim * 4; // Conventional for model architecture

        this.layerNorm1 = new RMSNorm(feature_dim);
        this.attention = new AttentionHeadSingle(feature_dim, head_dim);
        this.layerNorm2 = new RMSNorm(feature_dim);
        this.mlp = new Perceptron(feature_dim, hidden_dim);
    }

    forward(x) {
        // Attention sub layer, with residual
        const normalized1 = this.layerNorm1.forward(x);
        const attended = this.attention.forward(normalized1);
        const afterAttention = x.add(attended); // Residual add

        // MLP sub layer, with residual
        const normalized2 = this.layerNorm2.forward(afterAttention);
        const precieved = this.mlp.forward(normalized2);
        const afterMLP = afterAttention.add(precieved); // Residual add

        return afterMLP;
    }

    parameters() {
        const params = [];
        const components = [this.layerNorm1, this.attention, this.layerNorm2, this.mlp];
        for (let i = 0; i < components.length; ++i) {
            const sub = components[i].parameters();
            for (let j = 0; j < sub.length; ++j) {
                params.push(sub[j]);
            }
        }
        return params;
    }
}

class MinimalGPT { // GPTModel might have been a better name. Just noting for the future....
    constructor(vocabSize, featureDim, maxContextLength, numBlocks) {
        this.vocabSize = vocabSize;
        this.featureDim = featureDim;
        this.maxContextLength = maxContextLength;
        this.numBlocks = numBlocks;

        // Lookup tables. 
        // Row i of tokenEmbeddings = embedding vector for vocab token i.
        this.tokenEmbeddings = new Matrix(vocabSize, featureDim);
        this.tokenEmbeddings.initToSmallRandom();
        // Row i of positionalEmbeddings = embedding for position p.
        this.positionalEmbeddings = new Matrix(maxContextLength, featureDim);
        this.positionalEmbeddings.initToSmallRandom();

        // Stack of transformer blocks.
        this.blocks = [];
        for (let i = 0; i < numBlocks; ++i) {
            this.blocks.push(new TransformerBlock(featureDim));
        }

        // Final norm before unembedding.
        this.finalNorm = new RMSNorm(featureDim);
    }

    embed(tokenIdArray) {
        const numtokens = tokenIdArray.length;
        const embedded = new Matrix(numtokens, this.featureDim);
        // No init — every cell is filled with a freshly computed Value below.

        for (let row = 0; row < numtokens; ++row) {
            const tokenId = tokenIdArray[row];
            for (let col = 0; col < this.featureDim; ++col) {
                const index = row * this.featureDim + col;
                const tokenEmbedding = this.tokenEmbeddings.get(tokenId * this.featureDim + col);
                const positionalEmbedding = this.positionalEmbeddings.get(index);
                embedded.set(index, tokenEmbedding.add(positionalEmbedding));
            }
        }

        return embedded;
    }

    forward(tokenIdArray) {
        if (tokenIdArray.length > this.maxContextLength) {
            throw new Error(`input length ${tokenIdArray.length} exceeds maxContextLength ${this.maxContextLength}`);
        }

        let x = this.embed(tokenIdArray); // Shape: sequenceLength, featureDim
        for (let i = 0; i < this.blocks.length; ++i) {
            x = this.blocks[i].forward(x); // Shape preserved.
        }
        x = this.finalNorm.forward(x); // Shape: sequenceLength, featureDim

        // Tied unembedding: (sequenceLength, featureDim) * (featureDim, vocabSize)
        const logits = x.mul(this.tokenEmbeddings.transposed()); // sequenceLength, vocabSize
        return logits;
    }

    predictNextToken(tokenIdArray) {
        const logits = this.forward(tokenIdArray);
        const probabilities = logits.softMaxedRows();  
        // We don't need to soft-max here, doing it for compleatness

        // Only care about the last row of probabilities
        const lastRow = tokenIdArray.length - 1;
        const lastProbs = new Array(this.vocabSize);
        for (let i = 0; i < this.vocabSize; ++i) {
            lastProbs[i] = probabilities.get(lastRow * this.vocabSize + i);
        }

        // Argmax over probabilities.
        let bestId = 0;
        let bestScore = lastProbs[0].data;
        for (let i = 1; i < this.vocabSize; ++i) {
            if (lastProbs[i].data > bestScore) {
                bestScore = lastProbs[i].data;
                bestId = i;
            }
        }

        return bestId;
    }

    sampleNextToken(tokenIdArray, temperature = 0.8, topK = 20) {
        const logits = this.forward(tokenIdArray);
        const lastRow = tokenIdArray.length - 1;
        let candidates = new Array(this.vocabSize);

        for (let i = 0; i < this.vocabSize; ++i) {
            candidates[i] = {
                id: i,
                score: logits.get(lastRow * this.vocabSize + i).data
            };
        }

        if (temperature <= 0) {
            let best = candidates[0];
            for (let i = 1; i < candidates.length; ++i) {
                if (candidates[i].score > best.score) {
                    best = candidates[i];
                }
            }
            return best.id;
        }

        candidates.sort((a, b) => b.score - a.score);
        if (topK > 0 && topK < candidates.length) {
            candidates = candidates.slice(0, topK);
        }

        let maxScore = candidates[0].score;
        for (let i = 1; i < candidates.length; ++i) {
            if (candidates[i].score > maxScore) {
                maxScore = candidates[i].score;
            }
        }

        let total = 0;
        for (let i = 0; i < candidates.length; ++i) {
            const probability = Math.exp((candidates[i].score - maxScore) / temperature);
            candidates[i].probability = probability;
            total += probability;
        }

        let draw = Math.random() * total;
        for (let i = 0; i < candidates.length; ++i) {
            draw -= candidates[i].probability;
            if (draw <= 0) {
                return candidates[i].id;
            }
        }
        return candidates[candidates.length - 1].id;
    }

    generate(tokenizer, prompt, options = {}) {
        const temperature = options.temperature ?? 0.8;
        const topK = options.topK ?? 20;
        const maxNewTokens = options.maxNewTokens ?? this.maxContextLength;
        const stopText = options.stopText ?? "FIN.";
        const ids = tokenizer.encode(prompt);

        if (ids.length === 0) {
            throw new Error("prompt must encode to at least one token");
        }

        for (let step = 0; step < maxNewTokens; ++step) {
            if (ids.length >= this.maxContextLength) break;
            ids.push(this.sampleNextToken(ids, temperature, topK));

            const text = tokenizer.decode(ids);
            if (stopText && text.includes(stopText)) {
                return text;
            }
        }

        return tokenizer.decode(ids);
    }

    parameters() {
        const params = [];

        // Token embedding table (also used tied as the unembedding).
        const tokenParams = this.tokenEmbeddings.parameters();
        for (let i = 0; i < tokenParams.length; ++i) {
            params.push(tokenParams[i]);
        }

        // Positional embedding table.
        const posParams = this.positionalEmbeddings.parameters();
        for (let i = 0; i < posParams.length; ++i) {
            params.push(posParams[i]);
        }

        // Per-block parameters.
        for (let i = 0; i < this.blocks.length; ++i) {
            const blockParams = this.blocks[i].parameters();
            for (let j = 0; j < blockParams.length; ++j) {
                params.push(blockParams[j]);
            }
        }

        // Final norm gamma.
        const normParams = this.finalNorm.parameters();
        for (let i = 0; i < normParams.length; ++i) {
            params.push(normParams[i]);
        }

        return params;
    }

    serializeToArrayBuffer() {
        const params = this.parameters();
        const buffer = new ArrayBuffer(params.length * 4); // Float32 = 4 bytes
        const view = new Float32Array(buffer);
        for (let i = 0; i < params.length; ++i) {
            view[i] = params[i].data;
        }
        return buffer;
    }

    deserializeFromArrayBuffer(buffer) {
        const params = this.parameters();
        const view = new Float32Array(buffer);
        if (view.length !== params.length) {
            throw new Error(`weight count mismatch: file has ${view.length}, model expects ${params.length}`);
        }
        for (let i = 0; i < params.length; ++i) {
            params[i].data = view[i];
        }
    }
}

class SGDTrainer {
    constructor(model, learningRate) {
        this.model = model;
        this.learningRate = learningRate;
    }

    // Mean cross-entropy: -1/N * sum_i log(q[i, targetIds[i]])
    // where q = softmax(logits), one row per prediction.
    crossEntropyLoss(logits, targetIds) { 
        // logits shape:    Matrix(sequenceLength, vocabSize)   — Values
        // targetIds shape: Array(sequenceLength)               — plain integers
        const probs = logits.softMaxedRows(); // Matrix(sequenceLength, vocabSize)
        const sequenceLength = logits.rows;
        const vocabSize = logits.columns;

        if (targetIds.length !== sequenceLength) {
            throw new Error(`targets length ${targetIds.length} != logits rows ${sequenceLength}`);
        }

        // Seed with row 0, then chain .add — same pattern you use in matmul/softmax.
        let loss = probs.get(0 * vocabSize + targetIds[0]).log().neg();
        for (let i = 1; i < sequenceLength; ++i) {
            // row: i -> where we are in the sequence
            // col: targetIds[i] -> next token id in vocab (target id's are shifted)
            const targetProb = probs.get(i * vocabSize + targetIds[i]);
            loss = loss.add(targetProb.log().neg());
        }
        return loss.mul(1 / sequenceLength);
    }

    train(tokenIds) { // tokenIds shape: Array(N) of ints — one training sequence (fits into max context length)
        const N = tokenIds.length;
        if (N < 2) throw new Error(`need at least 2 tokens, got ${N}`);

        const inputIds = tokenIds.slice(0, N - 1); // Array(N-1) of ints, fed to model
        const targetIds = tokenIds.slice(1);       // Array(N-1) of ints, shifted by 1

        // Zero grads on parameters. Each forward builds a fresh graph
        const params = this.model.parameters();
        for (let i = 0; i < params.length; ++i) {
            params[i].grad = 0;
        }

        // Forward → loss → backward → step
        const logits = this.model.forward(inputIds);             // Matrix(N-1, vocabSize)
        const loss = this.crossEntropyLoss(logits, targetIds);   // single Value (scalar)
        loss.backward(); // populates .grad on every Value reachable from loss

        // Mutating .data directly bypasses the graph — intentional, the graph
        // is about to be GC'd and we don't want this update differentiated.
        for (let i = 0; i < params.length; ++i) {
            params[i].data -= this.learningRate * params[i].grad;
        }

        return loss.data;  // plain number for logging
    }
}

class Tokenizer {
    constructor() {
        // vocab[id] = Uint8Array of the raw bytes this token expands to.
        // Ids 0–255 are pre-seeded with single-byte values, so EVERY possible
        // UTF-8 byte sequence is encodable. No unknown tokens, ever.
        this.vocab = [];
        for (let i = 0; i < 256; ++i) {
            this.vocab.push(new Uint8Array([i]));
        }

        // merges: pairKey -> { rank, newId }
        //   pairKey  = a * 65536 + b  (cheap reversible packing for two ids)
        //   rank     = insertion order (lower rank = learned earlier = higher priority at encode time)
        //   newId    = the merged token's id in this.vocab
        this.merges = new Map();
    }

    get vocabSize() {
        return this.vocab.length;
    }

    // The regex-free version of GPT-2's pre-tokenizer.  We walk the string,
    // classify each character, and start a new chunk every time the category
    // changes.  Chunks are walls: BPE will count and merge pairs WITHIN a
    // chunk but never ACROSS chunk boundaries.  This is what stops BPE from
    // learning cross-word garbage tokens like " thecat".
    _classify(code) {
        // Tiny category function — a handful of ASCII ranges, no Unicode tables.
        if (code >= 0x30 && code <= 0x39) return 1;                         // digit  0–9
        if (code >= 0x41 && code <= 0x5A) return 2;                         // letter A–Z
        if (code >= 0x61 && code <= 0x7A) return 2;                         // letter a–z
        if (code === 0x20 || code === 0x09 || code === 0x0A || code === 0x0D) return 3; // whitespace
        if (code > 127) return 2;                                           // non-ASCII -> treat as letter-like
        return 4;                                                           // other (punctuation, symbols)
    }

    _preTokenize(text) {
        // Returns Array<Array<int>>: a list of chunks, each chunk a list of byte ids in 0–255.
        if (text.length === 0) return [];

        const chunks = [];
        let currentChars = [];
        let currentCat = this._classify(text.charCodeAt(0));

        for (let i = 0; i < text.length; ++i) {
            const cat = this._classify(text.charCodeAt(i));
            if (cat !== currentCat) {
                // Category boundary — emit the chunk we were building, start a new one.
                chunks.push(this._charsToBytes(currentChars));
                currentChars = [];
                currentCat = cat;
            }
            currentChars.push(text[i]);
        }
        chunks.push(this._charsToBytes(currentChars)); // trailing chunk

        return chunks;
    }

    _charsToBytes(chars) {
        // UTF-8 encode the chunk, then return an array of byte ids.
        // Since base vocab is identity over 0–255, the byte value IS the id.
        const bytes = new TextEncoder().encode(chars.join(''));
        const ids = new Array(bytes.length);
        for (let i = 0; i < bytes.length; ++i) {
            ids[i] = bytes[i];
        }
        return ids;
    }

    // Walk a chunk left-to-right, replacing every adjacent (a, b) with newId.
    // Greedy left-to-right is correct even for overlaps: [a,a,a] merging (a,a)
    // produces [X,a], not [a,X], which matches what the counter saw.
    _applyMerge(chunk, a, b, newId) {
        const result = [];
        for (let i = 0; i < chunk.length; ) {
            if (i < chunk.length - 1 && chunk[i] === a && chunk[i + 1] === b) {
                result.push(newId);
                i += 2;
            }
            else {
                result.push(chunk[i]);
                i += 1;
            }
        }
        return result;
    }

    train(text, targetVocabSize) {
        let corpus = this._preTokenize(text); // Array<Array<int>>, mutated in place across iterations

        while (this.vocab.length < targetVocabSize) {
            // 1. Global pair counts.  Only count within a chunk.
            const counts = new Map();
            for (let i = 0; i < corpus.length; ++i) {
                const chunk = corpus[i];
                for (let j = 0; j < chunk.length - 1; ++j) {
                    const key = chunk[j] * 65536 + chunk[j + 1];
                    counts.set(key, (counts.get(key) || 0) + 1);
                }
            }
            if (counts.size === 0) break; // corpus has no pairs left

            // 2. Argmax over counts.
            let bestKey = -1;
            let bestCount = 0;
            for (const [key, count] of counts) {
                if (count > bestCount) {
                    bestCount = count;
                    bestKey = key;
                }
            }
            // If even the winning pair only appears once, merging it just bloats
            // the vocab with a token that will match exactly one place. Stop.
            if (bestCount < 2) break;

            // 3. Record the merge and grow the vocab.
            const a = Math.floor(bestKey / 65536);
            const b = bestKey % 65536;
            const newId = this.vocab.length;
            this.merges.set(bestKey, { rank: this.merges.size, newId });

            // The new token's bytes = concat of its parents' bytes.  
            const aBytes = this.vocab[a];
            const bBytes = this.vocab[b];
            const merged = new Uint8Array(aBytes.length + bBytes.length);
            merged.set(aBytes, 0);
            merged.set(bBytes, aBytes.length);
            this.vocab.push(merged);

            // 4. Rewrite every chunk to use the new id wherever the pair appeared.
            for (let i = 0; i < corpus.length; ++i) {
                corpus[i] = this._applyMerge(corpus[i], a, b, newId);
            }
        }
    }

    encode(text) {
        const chunks = this._preTokenize(text);
        const result = [];

        for (let i = 0; i < chunks.length; ++i) {
            let chunk = chunks[i];

            while (true) {
                // Scan for the best (lowest-rank) merge currently present.
                let bestRank = Infinity;
                let bestA = -1;
                let bestB = -1;
                let bestNewId = -1;

                for (let j = 0; j < chunk.length - 1; ++j) {
                    const key = chunk[j] * 65536 + chunk[j + 1];
                    const merge = this.merges.get(key);
                    if (merge && merge.rank < bestRank) {
                        bestRank = merge.rank;
                        bestA = chunk[j];
                        bestB = chunk[j + 1];
                        bestNewId = merge.newId;
                    }
                }

                if (bestRank === Infinity) break; // no more applicable merges
                chunk = this._applyMerge(chunk, bestA, bestB, bestNewId);
            }

            for (let j = 0; j < chunk.length; ++j) {
                result.push(chunk[j]);
            }
        }

        return result;
    }

    decode(ids) {
        let totalLen = 0;
        for (let i = 0; i < ids.length; ++i) {
            totalLen += this.vocab[ids[i]].length;
        }

        const bytes = new Uint8Array(totalLen);
        let offset = 0;
        for (let i = 0; i < ids.length; ++i) {
            const tokBytes = this.vocab[ids[i]];
            bytes.set(tokBytes, offset);
            offset += tokBytes.length;
        }

        return new TextDecoder().decode(bytes);
    }

    serializeToJSON() {
        // Walk merges in rank order so replay produces identical state.
        const ordered = new Array(this.merges.size);
        for (const [key, info] of this.merges) {
            const a = Math.floor(key / 65536);
            const b = key % 65536;
            ordered[info.rank] = [a, b];
        }
        return JSON.stringify({ merges: ordered });
    }

    deserializeFromJSON(json) {
        const data = JSON.parse(json);

        // Reset to base state (matches constructor).
        this.vocab = [];
        for (let i = 0; i < 256; ++i) {
            this.vocab.push(new Uint8Array([i]));
        }
        this.merges = new Map();

        // Replay each merge — same logic as train(), minus the counting.
        for (let i = 0; i < data.merges.length; ++i) {
            const a = data.merges[i][0];
            const b = data.merges[i][1];
            const key = a * 65536 + b;
            const newId = this.vocab.length;
            this.merges.set(key, { rank: this.merges.size, newId });

            const aBytes = this.vocab[a];
            const bBytes = this.vocab[b];
            const merged = new Uint8Array(aBytes.length + bBytes.length);
            merged.set(aBytes, 0);
            merged.set(bBytes, aBytes.length);
            this.vocab.push(merged);
        }
    }
}

// Hyperparameters live here so both functions agree without sharing state.
// Change them in one place, retrain, the new files load cleanly.
const EXAMPLE_FEATURE_DIM = 72;
const EXAMPLE_MAX_CONTEXT = 96;
const EXAMPLE_NUM_BLOCKS = 3;

function _downloadBlob(data, filename, mimeType) {
    const blob = new Blob([data], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

function ExampleTrain() {
    const corpus =
        "Mr.pepper is in the study with a knife " +
        "Colnel Mustard is in the living room with a rope. " +
        "The butler is in the kitchen with a pipe. " +
        "A body was found in the hallway with stab wounds.  " +
        "Therefore the murderer is: Mr.pepper";

    // 1. Tokenizer.
    const tokenizer = new Tokenizer();
    tokenizer.train(corpus, 280); // 256 base bytes + up to 24 merges
    console.log(`tokenizer trained: vocabSize = ${tokenizer.vocabSize}`);

    // 2. Model. Tiny dims because every scalar is a Value graph node.
    const model = new MinimalGPT(
        tokenizer.vocabSize,
        EXAMPLE_FEATURE_DIM,
        EXAMPLE_MAX_CONTEXT,
        EXAMPLE_NUM_BLOCKS
    );

    // 3. Train.
    const sentences = [
        "Mr.pepper is in the study",
        "Colnel Mustard is in the living room",
        "The butler is in the kitchen",
    ];

    const trainer = new SGDTrainer(model, 0.1);
    const numEpochs = 30;

    for (let epoch = 0; epoch < numEpochs; ++epoch) {
        let epochLoss = 0;
        for (let i = 0; i < sentences.length; ++i) {
            const ids = tokenizer.encode(sentences[i]);
            epochLoss += trainer.train(ids);
        }
        epochLoss /= sentences.length;

        if (epoch % 5 === 0 || epoch === numEpochs - 1) {
            console.log(`epoch ${epoch}: avg loss = ${epochLoss.toFixed(4)}`);
        }
    }

    // 4. Save. Two separate downloads — your browser may prompt for each.
    _downloadBlob(model.serializeToArrayBuffer(), 'weights.bin', 'application/octet-stream');
    _downloadBlob(tokenizer.serializeToJSON(), 'vocab.json', 'application/json');
    console.log("training done — weights.bin and vocab.json downloaded");
}

// modelBuffer: ArrayBuffer (e.g. from file.arrayBuffer())
// vocabJson:   string       (e.g. from file.text())
function ExamplePredict(modelBuffer, vocabJson) {
    // 1. Rebuild tokenizer first — model construction needs its vocabSize.
    const tokenizer = new Tokenizer();
    tokenizer.deserializeFromJSON(vocabJson);
    console.log(`tokenizer loaded: vocabSize = ${tokenizer.vocabSize}`);

    // 2. Build model with the SAME hyperparameters used during training.
    //    These have to match or deserializeFromArrayBuffer will throw a size mismatch.
    const model = new MinimalGPT(
        tokenizer.vocabSize,
        EXAMPLE_FEATURE_DIM,
        EXAMPLE_MAX_CONTEXT,
        EXAMPLE_NUM_BLOCKS
    );
    model.deserializeFromArrayBuffer(modelBuffer);
    console.log("model loaded");

    // 3. Greedy autoregressive generation.
    const prompts = ["Mr.pepper", "Colnel Mustard", "The butler"];
    const maxNewTokens = 15;

    for (let p = 0; p < prompts.length; ++p) {
        const prompt = prompts[p];
        let ids = tokenizer.encode(prompt);

        console.log(`\nprompt:    "${prompt}"`);

        for (let step = 0; step < maxNewTokens; ++step) {
            if (ids.length >= model.maxContextLength) break;
            const nextId = model.predictNextToken(ids);
            ids.push(nextId);
        }

        console.log(`generated: "${tokenizer.decode(ids)}"`);
    }
}