import tensorflow as tf
import tensorflow.contrib as tc


def multihead_attention(memory, key_size, value_size, num_heads):
    # Perform linear tranformation to compute all Q, K, V
    qkv_size = 2 * key_size + value_size
    total_size = qkv_size * num_heads  # Denote as F.
    qkv = tf.layers.dense(memory, total_size)
    qkv = tc.layers.layer_norm(qkv)

    mem_slots = memory.get_shape().as_list()[1]  # Denoted as N.

    # [B, N, F] -> [B, N, H, F/H]
    qkv_reshape = tf.reshape(qkv, [-1, mem_slots, num_heads, qkv_size])

    # [B, N, H, F/H] -> [B, H, N, F/H]
    qkv_transpose = tf.transpose(qkv_reshape, [0, 2, 1, 3])
    q, k, v = tf.split(qkv_transpose, [key_size, key_size, value_size], -1)

    # softmax(QK^T/(d**2))V
    q *= key_size ** -0.5
    dot_product = tf.matmul(q, k, transpose_b=True)  # [B, H, N, N]
    weights = tf.nn.softmax(dot_product)
    output = tf.matmul(weights, v)  # [B, H, N, V]

    # [B, H, N, V] -> [B, N, H, V]
    output_transpose = tf.transpose(output, [0, 2, 1, 3])

    # [B, N, H, V] -> [B, N, H * V]
    new_memory = tf.reshape(output_transpose, [-1, mem_slots, num_heads * value_size])

    return new_memory

def initial_state(batch_size, mem_slots, mem_size):
    # [bach_size, mem_slots, mem_size]
    init_state = tf.eye(mem_slots, num_columns=mem_size, batch_shape=[batch_size])

    return init_state

def attend_over_memory(memory, key_size, value_size, num_heads, 
                       num_mlp_layers, num_blocks=1):
    def mlp(x, units_list):
        for u in units_list:
            x = tf.layers.dense(x, u)

        return x

    mem_size = num_heads * value_size
    for _ in range(num_blocks):
        attended_memory = multihead_attention(memory, key_size, value_size, num_heads)
        # Add a skip connection to the multiheaded attention's input.
        memory = tc.layers.layer_norm(memory + attended_memory)

        mlp_memory = mlp(memory, [mem_size] * num_mlp_layers)
        # Add a skip connection to the attention_mlp's input.
        memory = tc.layers.layer_norm(memory + mlp_memory)

    return memory

def create_gates(inputs, memory, mem_size, gate_style): 
    def calculate_gate_size(mem_size, gate_style):
        if gate_style == 'unit':
            return mem_size
        elif gate_style == 'memory':
            return 1
        else:  # gate_style == None
            return 0
    # We'll create the input and forget gates at once.
    # Hence, calculate double the gate size.
    num_gates = 2 * calculate_gate_size(mem_size, gate_style)

    memory = tf.tanh(memory)
    # Do not take the following code as
    # split(dense(concat([inputs, memory], 2), num_gates), 2, 2)
    # They are different since inputs and memory have different dimension at axis=1
    # In this sense, they are more like
    # split(dense(concat([inputs + zeros_like(memory), memory], 2), num_gates), 2, 2)
    gate_inputs = tf.layers.dense(inputs, num_gates)
    gate_memory = tf.layers.dense(memory, num_gates)
    gates = tf.split(gate_memory + gate_inputs, num_or_size_splits=2, axis=2)
    input_gate, forget_gate = gates

    # There is bias terms inside sigmoid in the original implementation, 
    # which I omit for simplicity here
    input_gate = tf.sigmoid(input_gate)
    forget_gate = tf.sigmoid(forget_gate)

    return input_gate, forget_gate

def RMC(inputs, memory, key_size, value_size, num_heads,
        num_mlp_layers, num_blocks=1, gate_style='unit'):
    mem_size = num_heads * value_size
    inputs = tf.layers.dense(inputs, mem_size)
    if len(inputs.shape.as_list()) == 2:
        # reshape inputs so as to be ready to connect to memory
        inputs_reshape = inputs[:, None]
    # Inputs shape: [B, N_i, F]
    # Memory shape: [B, N_m, F]
    
    # Memory_plus_input shape: [B, N_m + N_i, F]
    memory_plus_input = tf.concat([memory, inputs_reshape], axis=1)
    # Next memory shape: [B, N_m + N_i, F]
    next_memory = attend_over_memory(memory_plus_input, key_size, value_size, num_heads, num_mlp_layers, num_blocks)
    n = inputs_reshape.get_shape().as_list()[1]
    # Crop next_memory to restore shape to [B, N_m, F]
    next_memory = next_memory[:, :-n, :]

    if gate_style == 'unit' or gate_style == 'memory':
        input_gate, forget_gate = create_gates(
          inputs_reshape, memory, mem_size, gate_style)
        next_memory = input_gate * tf.tanh(next_memory)
        next_memory += forget_gate * memory

    # Output shape: [B, N_m * F]
    output = tf.reshape(next_memory, [-1, n * mem_size])
    
    return output, next_memory