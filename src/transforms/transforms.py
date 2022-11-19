from torchvision import transforms


train_transforms = transforms.Compose([
                                    transforms.RandomRotation(degrees=30),
                                    transforms.CenterCrop(256)
                                   ])

valid_transforms = transforms.Compose([
                                    transforms.CenterCrop(256)
                                   ])

test_transforms = transforms.Compose([
                                transforms.CenterCrop(256)
                               ])
