import torch
from thop import profile, clever_format
from Model import LegoPartsModelV1

def print_layer_info(layer_dict, indent=0):
    for layer_name, (ops, params, children) in layer_dict.items():
        ops_str, params_str = clever_format([ops, params], "%.3f")
        print("  " * indent + f"{layer_name}: MACs={ops_str}, Params={params_str}")
        if children:
            print_layer_info(children, indent + 1)

# Use a small output_shape example here.
model = LegoPartsModelV1(input_shape=3, hidden_units=32, output_shape=8)
dummy_input = torch.randn(1, 3, 128, 128)

macs, params, layer_info = profile(
    model,
    inputs=(dummy_input,),
    ret_layer_info=True,
    verbose=False
)

total_macs, total_params = clever_format([macs, params], "%.3f")
print(f"Total MACs: {total_macs}")
print(f"Total Params: {total_params}")
print("\nPer-layer breakdown:")
print_layer_info(layer_info)