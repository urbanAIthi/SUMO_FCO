def test(model, test_IDs, analyse = False, analyse_path = 'test'):
    if analyse:
        if not os.path.exists(analyse_path):
            os.makedirs(analyse_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    test_acc = list()
    test_pre = list()
    test_rec = list()
    output_label_comps = ('TP', 'FP', 'TN', 'FN')
    for olc in output_label_comps:
        if os.path.exists(os.path.join(analyse_path, olc)):
            shutil.rmtree(os.path.join(analyse_path, olc))
        os.mkdir(os.path.join(analyse_path, olc))
    for test_ID in test_IDs:
        test_set = Dataset(test_ID, transforms)
        test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
        with torch.no_grad():
            for images, vectors, labels, names in iter(test_loader):
                images, vectors, labels = images.to(device), vectors.to(device), labels.to(device)

                # Perform the same preprocessing as in the training loop, if necessary

                outputs = model(images.float(), vectors.float())
                labels = labels.unsqueeze(1).float()

                accuracy, precision, recall, _ = binary_metrics(outputs, labels)
                if analyse:
                    for i in range(len(images)):
                        output_label_comp = get_single_classification_metric(outputs[i], labels[i])
                        img = images[i].cpu().numpy()
                        img_pil = PIL.Image.fromarray(np.uint8(img.transpose(1, 2, 0)))
                        #draw circle for ego vehicle
                        img_pil = draw_circle_at_position(img_pil, 5, 0, 0, (255, 0, 0))
                        #draw circle at vector position
                        img_pil = draw_circle_at_position(img_pil, 5, vectors[i][0], vectors[i][1], (0, 255, 0))
                        img_pil.save(os.path.join(analyse_path, output_label_comp, str(names[i]) + '.jpg'))
                test_acc.append(accuracy)
                test_pre.append(precision)
                test_rec.append(recall)

        average_test_acc = sum(test_acc) / len(test_acc)
        average_test_pre = sum(test_pre) / len(test_pre)
        average_test_rec = sum(test_rec) / len(test_rec)
        print('Test accuracy: ', average_test_acc, 'Test precision: ', average_test_pre, 'Test recall: ', average_test_rec)
        wandb.log({"test_acc": average_test_acc})
        wandb.log({"test_pre": average_test_pre})
        wandb.log({"test_rec": average_test_rec})
        shutil.make_archive('test_images', 'zip', 'test')
        wandb.save('test_images.zip')

def binary_metrics(predictions, targets, threshold=0.5):
    # Convert probabilities to binary predictions
    binary_preds = (predictions >= threshold).float()
    count_ones = torch.sum(binary_preds == 1).item()

    # Calculate true positives, false positives, and false negatives
    true_positives = (binary_preds * targets).sum()
    false_positives = (binary_preds * (1 - targets)).sum()
    false_negatives = ((1 - binary_preds) * targets).sum()

    # Calculate accuracy, precision, and recall
    accuracy = (true_positives + (1 - binary_preds).sum() - false_negatives) / len(targets)
    precision = true_positives / (true_positives + false_positives + 1e-9)  # Add small epsilon to avoid division by zero
    recall = true_positives / (true_positives + false_negatives + 1e-9)  # Add small epsilon to avoid division by zero

    return accuracy, precision, recall, count_ones

def get_single_classification_metric(output, label, threshold=0.5):
    binary_prediction = (output > threshold).float()

    if binary_prediction == 1 and label == 1:
        return "TP"
    elif binary_prediction == 1 and label == 0:
        return "FP"
    elif binary_prediction == 0 and label == 0:
        return "TN"
    elif binary_prediction == 0 and label == 1:
        return "FN"

def draw_circle_at_position(img, circle_radius, center_x=0, center_y=0, circle_color=(255, 0, 0), circle_width=1, mtopixel = 0.4):
    img_copy = img.copy()
    width, height = img.size

    # Transform the input coordinates to use the center of the image as the origin
    transformed_x = width // 2
    transformed_y = height // 2

    transformed_x = transformed_x + ((1/mtopixel) * center_x)
    transformed_y = transformed_y - ((1/mtopixel) * center_y) #minus because of the different coordinate system

    draw = ImageDraw.Draw(img_copy)
    draw.ellipse((transformed_x - circle_radius, transformed_y - circle_radius, transformed_x + circle_radius, transformed_y + circle_radius), outline=circle_color, width=circle_width)
    return img_copy