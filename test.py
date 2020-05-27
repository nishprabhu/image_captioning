""" Unit Test """


def main():
    """ Main function """
    root = "coco_dataset/train2017"
    ann_file = "coco_dataset/annotations/captions_train2017.json"
    dataset = Dataset(root, ann_file)
    dataloader = DataLoader(dataset, batch_size=10, collate_fn=collate_fn)

    vocab_size = dataset.tokenizer.vocab_size
    embedding_dim = 300
    encoder_output_dim = 120

    model = CaptioningModel(vocab_size, embedding_dim, encoder_output_dim)

    for batch in dataloader:
        images, captions = batch
        print(images.shape)
        print(captions.shape)

        output = model(images, captions, predict=True)
        print(output.shape)
        break


if __name__ == "__main__":
    main()
