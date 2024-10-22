import torch
from monai.transforms import EnsureChannelFirstD

def test_ensure_channel_d():
    # Initialize the EnsureChannelD transform
    # Specify the keys to which the transform should be applied
    # Here, we'll assume the data is under the key 'image'
    ensure_channel = EnsureChannelFirstD(keys=['image'])

    # Create sample data
    # Case 1: Input tensor already has a channel dimension (C, H, W)
    tensor_with_channel = torch.randn(3, 224, 224)  # Example with 3 channels

    # Case 2: Input tensor without a channel dimension (H, W)
    tensor_without_channel = torch.randn(224, 224)  # Example with no channels

    # Wrap tensors in a dictionary as MONAI's dictionary-based transforms expect dict inputs
    data_with_channel = {'image': tensor_with_channel}
    data_without_channel = {'image': tensor_without_channel}

    # Apply the EnsureChannelD transform
    # transformed_with_channel = ensure_channel(data_with_channel)
    transformed_without_channel = ensure_channel(data_without_channel)

    # Retrieve the transformed tensors
    # transformed_tensor_with_channel = transformed_with_channel['image']
    transformed_tensor_without_channel = transformed_without_channel['image']

    # Print shapes to verify
    print("=== Testing EnsureChannelD ===\n")

    # print("Case 1: Input tensor already has a channel dimension (C, H, W)")
    # print(f"Original shape: {tensor_with_channel.shape}")
    # print(f"Transformed shape: {transformed_tensor_with_channel.shape}\n")

    print("Case 2: Input tensor without a channel dimension (H, W)")
    print(f"Original shape: {tensor_without_channel.shape}")
    print(f"Transformed shape: {transformed_tensor_without_channel.shape}\n")

    # Additional Assertions (optional)
    # assert transformed_tensor_with_channel.shape == tensor_with_channel.shape, \
    #     "EnsureChannelD should not alter tensors that already have the channel dimension."

    assert transformed_tensor_without_channel.shape[0] == 1, \
        "EnsureChannelD should add a channel dimension to tensors without one."

    print("All tests passed successfully!")

if __name__ == "__main__":
    test_ensure_channel_d()
