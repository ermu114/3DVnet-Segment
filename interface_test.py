import interface

if __name__ == '__main__':
    model_root_path = 'D:\\tmp\\MIS\\model'
    interface.initialize(model_root_path)
    one_case_path = 'D:\\tmp\\MIS\\test\\1'
    modality = 'CT'
    interface.predict(one_case_path, modality)

    print('end')