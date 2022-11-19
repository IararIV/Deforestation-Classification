from torchvision import transforms

# IMAGES
# mean: [0.06, 0.12, 0.08]
# std:  [0.04, 0.04, 0.05]

train_transforms = transforms.Compose([
                                    transforms.RandomVerticalFlip(0.5),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.RandomCrop(224),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])

valid_transforms = transforms.Compose([
                                    transforms.RandomCrop(224),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])

test_transforms = transforms.Compose([
                                transforms.RandomCrop(224),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])
