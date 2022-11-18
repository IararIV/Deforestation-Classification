from torchvision import transforms


train_transforms = transforms.Compose([
                                    transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                                    transforms.RandomRotation(degrees=15),
                                    transforms.CenterCrop(224),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])

valid_transforms = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])

test_transforms = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])
