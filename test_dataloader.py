import sys
from args.arg_parse import ArgParse
from dataset.dataset_pl import TrainValDataModule


def test():

    print("=== Dataloader Test ===")

    # ArgParse.get() を呼び出すとコマンドラインから引数を受け取れます
    args = ArgParse.get()

    print(f"\n[Info] Target Dataset: {args.dataset_name}")
    if args.dataset_name == "ImageFolder":
        print("[Warning] ImageFolderは現在のsequential_video_datasetではサポートされていません。UCFや50Salads等を指定してください。")

    try:
        data_module = TrainValDataModule(
            command_line_args=args,
            dataset_name=args.dataset_name
        )

        train_loader = data_module.train_dataloader()
        print(f"[Info] Num classes: {data_module.n_classes}")

        print("\n[Info] Loading first train batch...")
        # 最初のバッチを取得してループを抜ける
        for i, batch in enumerate(train_loader):
            print("\n:white_check_mark: Successfully loaded batch!")

            # batch の中身を確認
            if isinstance(batch, (list, tuple)):
                print(f"Batch has {len(batch)} elements (defined in collate_fn).")
                for j, item in enumerate(batch):
                    if hasattr(item, "shape"):
                        print(f"  element {j} shape: {item.shape}")
                    elif isinstance(item, list):
                        if len(item) > 0 and hasattr(item[0], "shape"):
                            print(f"  element {j} is list. length: {len(item)}, item[0] shape: {item[0].shape}")
                        else:
                            print(f"  element {j} is list. length: {len(item)}")
                    else:
                        print(f"  element {j} type: {type(item)}")
            else:
                print(f"Batch shape (or type): {type(batch)}")

            break  # 1バッチだけでテスト終了

    except Exception as e:
        import traceback
        print("\n:x: Error occurred while loading data:")
        traceback.print_exc()


if __name__ == "__main__":
    test()
