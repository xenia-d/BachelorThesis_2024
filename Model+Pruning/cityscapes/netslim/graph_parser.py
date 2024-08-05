import re
import torch

CHANNEL_DIM = 1
NORM_LAYER_KEYWORDS = ["batch_norm", "group_norm", "instance_norm"]
PRUNABLE_LAYER_KEYWORDS = ["convolution", "addmm"]  # does not support groups > 1 for conv
PASS_KEYWORDS = ["relu", "leaky_relu", "sigmoid", "tanh",
                 "pool", "pad", "dropout",
                 "view", "flatten"]  # and more .. does not support concat
OTHER_OP_KEYWORDS = ["cat"]
OTHER_PRIM_KEYWORDS = ["ListConstruct"]  # for cat
NO_EFFECT_KEYWORDS = ["size", ]


scope_pattern = re.compile(r".+, scope: (.+)")
module_pattern = re.compile(r"\[(\w+)\]")
output_pattern = re.compile(r"(%.+) : .+ = .+")
input_pattern = re.compile(r".+ = aten::\w+\((%.+),*\)")
prim_input_pattern = re.compile(r".+ = prim::\w+\((%.+),*\)")
shape_pattern = re.compile(r"%.+ : \w+\((.+)\) = aten::\w+\(%.+\)")
int_pattern = re.compile(r"[1-9]+")
view_pattern = re.compile(r"aten::.*view.*")

norm_layer_pattern = re.compile(
    r".*= ({})\(.*\)".format(
        '|'.join(["aten::\w*{}\w*".format(_) for _ in NORM_LAYER_KEYWORDS])
    )
)

prunable_layer_pattern = re.compile(
    r".*= ({})\(.*\)".format(
        '|'.join(["aten::\w*{}\w*".format(_) for _ in PRUNABLE_LAYER_KEYWORDS])
    )
)

pass_layer_pattern = re.compile(
    r".*= ({})\(.*\)".format(
        '|'.join(["aten::\w*{}\w*".format(_) for _ in PASS_KEYWORDS])
    )
)

allowed_layer_pattern = re.compile(
    r".*= ({})\(.*\)".format(
        '|'.join(["aten::\w*{}\w*".format(_) for _ in
                  PRUNABLE_LAYER_KEYWORDS +
                  PASS_KEYWORDS +
                  NO_EFFECT_KEYWORDS])
    )
)

common_layer_pattern = re.compile(
    r".*= ({})\(.*\)".format(
        '|'.join(["aten::\w*{}\w*".format(_) for _ in
                  NORM_LAYER_KEYWORDS +
                  PRUNABLE_LAYER_KEYWORDS +
                  PASS_KEYWORDS +
                  OTHER_OP_KEYWORDS]
                 +["prim::\w*{}\w*".format(_) for _ in
                   OTHER_PRIM_KEYWORDS])
    )
)

tensor_op_pattern = re.compile(r"(%.+) : \w+\(.+\) = aten::\w+\(.*\)")

add_op_pattern = re.compile(r".*= (aten::add_)\(.*\)")

def get_node_str(node):
    return repr(node).split(" # ")[0]


# def parse_module_name(x):
#     scope_found = scope_pattern.findall(x)
#     module_name = ''
#     #tokens = scope_found[0].split('/')[1:]
#     print("Tokens Length")
#     #for token in tokens:
#         #print(token)
#     #module_name = '.'.join([module_pattern.findall(_)[0] for _ in tokens])
#     return module_name

def parse_module_name(x):
    # Split the node string by whitespace to extract the operation name
    tokens = x.split()
    
    # Iterate through the tokens to find the operation name
    for token in tokens:
        # Check if the token matches the pattern of an operation name
        if token.startswith("aten::") or token.startswith("prim::"):
            # Extract the operation name by removing the prefix
            operation_name = token.split("::")[-1]
            return operation_name
    
    # If no operation name is found, return a default value or handle as needed
    return "unknown_module"


def parse_output_name(x):
    return output_pattern.findall(x)[0]


def parse_input_names(x):
    result = input_pattern.findall(x)
    if not result:
        result = prim_input_pattern.findall(x)
    return result[0].split(", ")


def parse_output_shape(x):
    sizes = shape_pattern.findall(x)[0].split(", ")
    for s in sizes:
        if not int_pattern.match(s):
            return None
    return [int(_) for _ in sizes]


# assume for a normalization layer, it has only one input/output
def get_norm_layer_io(graph):
    out2nl = {}
    in2nl = {}
    bn_names = []
    for node in graph.nodes():
        node_str = get_node_str(node)
        if norm_layer_pattern.match(node_str):
            bn_name = parse_module_name(node_str)
            output = parse_output_name(node_str)
            input = parse_input_names(node_str)[0]
            out2nl[output] = bn_name
            in2nl[input] = bn_name
            bn_names.append(bn_name)
    return out2nl, in2nl, bn_names


def reverse_search_dict(val, target_dict):
    return [k for k, v in target_dict.items() if v == val]


# check for tensor operation layer and prim::ListConstruct, which is used by cat operation
def get_input_count(graph):
    input_count = {}
    for node in graph.nodes():
        node_str = get_node_str(node)
        matches = common_layer_pattern.findall(node_str)
        if matches:
            input_names = parse_input_names(node_str)
            for input_name in input_names:
                if input_name not in input_count:
                    input_count[input_name] = 1
                else:
                    input_count[input_name] += 1
    return input_count


def get_pruning_layers(model, input_shape, device=None):
    """Parse the model trace graph and generate mapping to BNs

    Arguments:
        model (pytorch model): The model instance
        input_shape (tuple): Shape of the input tensor

    Returns:
        prec_layers (dict): Mapping from BN names to preceding convs/linears
        succ_layers (dict): Mapping from BN names to succeeding convs/linears
    """

    prunable_layers = []
    batchnorm_layers = []

    for name, module in model.named_modules():
        # Check if the module itself is a Conv2d layer
        if isinstance(module, torch.nn.Conv2d):
            prunable_layers.append(name)

        # Check if the module itself is a BatchNorm2d layer
        if isinstance(module, torch.nn.BatchNorm2d):
            batchnorm_layers.append(name)

  
    prec_layers = {}
    succ_layers = {}

    # Iterate through BatchNorm layers



    for i, bn_layer_name in enumerate(batchnorm_layers):
        # Extract the preceding Conv2d layer name
        prec_conv_name = prunable_layers[i] if i >= 0 else None

        # Extract the succeeding Conv2d layer name if it exists
        if i < len(batchnorm_layers) - 1:
            succ_conv_name = prunable_layers[i + 1]
        else:
            succ_conv_name = None

        # Add mappings to dictionaries
        prec_layers[bn_layer_name] = prec_conv_name
        succ_layers[bn_layer_name] = succ_conv_name

   

    return prunable_layers, batchnorm_layers, prec_layers, succ_layers

def get_norm_layer_names(model, input_shape):
    prec_layers, succ_layers, norm_layer_names = get_pruning_layers(model, input_shape)
    prunable_norm_layer_names = list(set(succ_layers) & set(prec_layers))
    maskable_norm_layer_names = list(set(succ_layers) - set(prec_layers))
    return prunable_norm_layer_names, maskable_norm_layer_names, norm_layer_names
