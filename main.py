
def execute_pipeline_salad():
    from pipelines.pipeline_salad import main_pipeline_wb
    from config_reader_utls import attrDict, config_reader_utls

    config_file_path = 'config/config_salad_wb.yaml'
    args_ = attrDict.AttrDict.from_nested_dicts(config_reader_utls.read_file(file_path=config_file_path))
    main_pipeline_wb(args=args_)

if __name__ == '__main__':
    execute_pipeline_salad()