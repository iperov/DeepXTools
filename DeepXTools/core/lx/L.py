import re

allowed_langs = {'en' : 'English',
                 'ru' : 'Русский',
                 }

_id_to_string_dict = {

    ##########
    ### COMMON
    ##########
    'New':{
        'en' : 'New',
        'ru' : 'Новый',
    },

    'Load':{
        'en' : 'Load',
        'ru' : 'Загрузить',
    },

    'Open':{
        'en' : 'Open',
        'ru' : 'Открыть',
    },

    'Save':{
        'en' : 'Save',
        'ru' : 'Сохранить',
    },
    
    'Change':{
        'en' : 'Change',
        'ru' : 'Изменить',
    },

    'Close':{
        'en' : 'Close',
        'ru' : 'Закрыть',
    },

    'Error':{
        'en' : 'Error',
        'ru' : 'Ошибка',
    },
    
    'Success':{
        'en' : 'Success',
        'ru' : 'Успешно',
    },
    
    'Info':{
        'en' : 'Info',
        'ru' : 'Инфо',
    },
    
    'Ok':{
        'en' : 'Ok',
        'ru' : 'Ок',
    },

    'Cancel' : {
        'en' : 'Cancel',
        'ru' : 'Отмена',
    },

    'Mode' : {
        'en' : 'Mode',
        'ru' : 'Режим',
    },

    'Reset' : {
        'en' : 'Reset',
        'ru' : 'Сброс',
    },

    'Hold':{
        'en' : 'Hold',
        'ru' : 'Удерживать',
    },

    'Train' : {
        'en' : 'Train',
        'ru' : 'Тренировать',
    },

    'Reveal_in_explorer':{
        'en' : 'Reveal in explorer',
        'ru' : 'Открыть в проводнике',
    },

    'Add_item':{
        'en' : 'Add item',
        'ru' : 'Добавить позицию',
    },

    'Remove_item':{
        'en' : 'Remove item',
        'ru' : 'Убрать позицию',
    },
    ############
    ### Specific
    ############

    'QAppWindow.Application':{
        'en' : 'Application',
        'ru' : 'Приложение',
    },

    'QAppWindow.Language':{
        'en' : 'Language',
        'ru' : 'Язык',
    },

    'QAppWindow.Help':{
        'en' : 'Help',
        'ru' : 'Помощь',
    },

    'QAppWindow.About':{
        'en' : 'About',
        'ru' : 'О программе',
    },

    'QAppWindow.Process_priority':{
        'en' : 'Processs priority',
        'ru' : 'Приоритет процесса',
    },

    'QAppWindow.Process_priority.Normal':{
        'en' : 'Normal',
        'ru' : 'Нормальный',
    },
    'QAppWindow.Process_priority.Lowest':{
        'en' : 'Lowest',
        'ru' : 'Самый низкий',
    },

    'QAppWindow.Reset_UI_settings':{
        'en' : 'Reset UI settings (requires restart)',
        'ru' : 'Сброс настроек интерфейса (требуется перезапуск)',
    },

    'QAppWindow.AsyncX_monitor':{
        'en' : 'AsyncX monitor',
        'ru' : 'AsyncX монитор',
    },
    
    'QAppWindow.Show_console':{
        'en' : 'Show console',
        'ru' : 'Показать консоль',
    },
    
    'QAppWindow.Quit':{
        'en' : 'Quit',
        'ru' : 'Выход',
    },

    'QxFileStateManager.Backup':{
        'en' : 'Backup',
        'ru' : 'Резервная копия',
    },

    'QxFileStateManager.Save.Every':{
        'en' : 'Every',
        'ru' : 'Каждые',
    },
    'QxFileStateManager.Save.minutes':{
        'en' : 'minutes',
        'ru' : 'минут',
    },

    'QxFileStateManager.Save.Maximum':{
        'en' : 'Maximum',
        'ru' : 'Максимум',
    },

    'QxFileStateManager.Save.backups':{
        'en' : 'backups',
        'ru' : 'резервных копий',
    },

    'QxFileStateManager.Notes':{
        'en' : 'Notes',
        'ru' : 'Заметки',
    },

    'QxGraph.Average_for':{
        'en' : 'Average for',
        'ru' : 'Среднее для',
    },

    'QxDeepRoto.File_state_manager':{
        'en' : 'File state manager',
        'ru' : 'Менеджер файлового состояния',
    },

    'QxDeepRoto.Data_generator':{
        'en' : 'Data generator',
        'ru' : 'Генератор данных',
    },

    'QxDeepRoto.Model':{
        'en' : 'Model',
        'ru' : 'Модель',
    },

    'QxDeepRoto.Trainer':{
        'en' : 'Trainer',
        'ru' : 'Тренер',
    },

    'QxDeepRoto.Export' : {
        'en' : 'Export',
        'ru' : 'Экспорт',
    },

    'QxDataGenerator.Reload' : {
        'en' : 'Reload',
        'ru' : 'Перезагрузить',
    },

    'QxDataGenerator.Mode.Fit' : {
        'en' : 'Fit',
        'ru' : 'Вместить',
    },

    'QxDataGenerator.Mode.Patch' : {
        'en' : 'Patch',
        'ru' : 'Патч',
    },

    'QxDataGenerator.Offset' : {
        'en' : 'Offset',
        'ru' : 'Смещение',
    },

    'QxDataGenerator.Random' : {
        'en' : 'Random',
        'ru' : 'Случайно',
    },

    'QxDataGenerator.Translation_X' : {
        'en' : 'Translation-X',
        'ru' : 'Перенос по X',
    },

    'QxDataGenerator.Translation_Y' : {
        'en' : 'Translation-Y',
        'ru' : 'Перенос по Y',
    },

    'QxDataGenerator.Scale' : {
        'en' : 'Scale',
        'ru' : 'Масштаб',
    },

    'QxDataGenerator.Rotation' : {
        'en' : 'Rotation',
        'ru' : 'Поворот',
    },

    'QxDataGenerator.Transform_intensity' : {
        'en' : 'Transform intensity',
        'ru' : 'Интенсивность транформации',
    },

    'QxDataGenerator.Image_deform_intensity' : {
        'en' : 'Image deform intensity',
        'ru' : 'Интенсивность деформации изображения',
    },

    'QxDataGenerator.Mask_deform_intensity' : {
        'en' : 'Mask deform intensity',
        'ru' : 'Интенсивность деформации маски',
    },

    'QxDataGenerator.Flip' : {
        'en' : 'Flip',
        'ru' : 'Отразить',
    },

    'QxDataGenerator.Levels_shift' : {
        'en' : 'Levels shift',
        'ru' : 'Смещение уровней',
    },

    'QxDataGenerator.Sharpen_blur' : {
        'en' : 'Sharpen/blur',
        'ru' : 'Резкость/размытие',
    },

    'QxDataGenerator.Glow_shade' : {
        'en' : 'Glow/shade',
        'ru' : 'Блики/тени',
    },

    'QxDataGenerator.Resize' : {
        'en' : 'Resize',
        'ru' : 'Пережатие',
    },

    'QxDataGenerator.JPEG_artifacts' : {
        'en' : 'JPEG artifacts',
        'ru' : 'JPEG артефакты',
    },

    'QxDataGenerator.Output_type' : {
        'en' : 'Output type',
        'ru' : 'Выходной тип',
    },
    
    'QxDataGenerator.Image_n_Mask' : {
        'en' : 'Image and mask',
        'ru' : 'Изображение и маска',
    },
    
    'QxDataGenerator.Image_n_ImageGrayscaled' : {
        'en' : 'Image and Image grayscaled',
        'ru' : 'Изображение и обесцвеченное изображение',
    },
    
    'QxDataGenerator.Generate_preview' : {
        'en' : 'Generate preview',
        'ru' : 'Генерировать предпросмотр',
    },

    'QxDataGenerator.Image' : {
        'en' : 'Image',
        'ru' : 'Изображение',
    },

    'QxDataGenerator.Mask' : {
        'en' : 'Mask',
        'ru' : 'Маска',
    },
    
    'MxModel.Exporting_model_to' : {
        'en' : 'Exporting model to',
        'ru' : 'Экспортируем модель в',
    },
    
    'MxModel.Importing_model_from' : {
        'en' : 'Importing model from',
        'ru' : 'Импортируем модель из',
    },
    
    'MxModel.Downloading_model_from' : {
        'en' : 'Downloading model from',
        'ru' : 'Скачиваем модель из',
    },
    
    'QxModel.Device' : {
        'en' : 'Device',
        'ru' : 'Устройство',
    },

    'QxModel.Resolution' : {
        'en' : 'Resolution',
        'ru' : 'Разрешение',
    },

    'QxModel.Base_dimension' : {
        'en' : 'Base dimension',
        'ru' : 'Базовая размерность',
    },
    
    'QxModel.UNet_mode' : {
        'en' : 'U-Net mode',
        'ru' : 'U-Net режим',
    },
    
    'QxModel.Input' : {
        'en' : 'Input',
        'ru' : 'Вход',
    },

    'QxModel.InputType.Color' : {
        'en' : 'Color',
        'ru' : 'Цвет',
    },

    'QxModel.InputType.Luminance' : {
        'en' : 'Luminance',
        'ru' : 'Яркость',
    },

    'QxModel.Current_settings' : {
        'en' : 'Current settings',
        'ru' : 'Текущие настройки',
    },

    'QxModel.Apply_settings' : {
        'en' : 'Apply settings',
        'ru' : 'Применить настройки',
    },
    
    'QxModel.Reset_model' : {
        'en' : 'Reset model',
        'ru' : 'Сбросить модель',
    },
    
    'QxModel.Import_model' : {
        'en' : 'Import model',
        'ru' : 'Импорт модели',
    },
    
    'QxModel.Export_model' : {
        'en' : 'Export model',
        'ru' : 'Экспорт модели',
    },
    
    'QxModel.Download_pretrained_model' : {
        'en' : 'Download pretrained model',
        'ru' : 'Скачать предтренированную модель',
    },
    
    'QxModelTrainer.Batch_size' : {
        'en' : 'Batch size',
        'ru' : 'Размер батча',
    },
    
    'QxModelTrainer.Batch_acc' : {
        'en' : 'Batch accumulation',
        'ru' : 'Аккумуляция батча',
    },
    
    'QxModelTrainer.Learning_rate' : {
        'en' : 'Learning rate',
        'ru' : 'Скорость обучения',
    },

    'QxModelTrainer.power' : {
        'en' : 'power',
        'ru' : 'сила',
    },

    'QxModelTrainer.Iteration_time' : {
        'en' : 'Iteration time',
        'ru' : 'Время итерации',
    },

    'QxModelTrainer.second' : {
        'en' : 'second',
        'ru' : 'секунд',
    },

    'QxModelTrainer.Start_training' : {
        'en' : 'Start training',
        'ru' : 'Начать тренировку',
    },

    'QxModelTrainer.Stop_training' : {
        'en' : 'Stop training',
        'ru' : 'Остановить тренировку',
    },

    'QxModelTrainer.Metrics' : {
        'en' : 'Metrics',
        'ru' : 'Метрики',
    },

    'QxExport.Input' : {
        'en' : 'Input',
        'ru' : 'Вход',
    },

    'QxExport.Output' : {
        'en' : 'Output',
        'ru' : 'Выход',
    },

    'QxExport.Patch_mode' : {
        'en' : 'Patch mode',
        'ru' : 'Режим патча',
    },

    'QxExport.Export' : {
        'en' : 'Export',
        'ru' : 'Экспортировать',
    },

    'QxExport.Sample_count' : {
        'en' : 'Sample count',
        'ru' : 'Кол-во семплов',
    },
    
    'QxExport.Fix_borders' : {
        'en' : 'Fix borders',
        'ru' : 'Фикс границ',
    },
    
    'QxPreview.Source' : {
        'en' : 'Source',
        'ru' : 'Источник',
    },

    'QxPreview.Data_generator' : {
        'en' : 'Data generator',
        'ru' : 'Генератор данных',
    },

    'QxPreview.Directory':{
        'en' : 'Directory',
        'ru' : 'Директория',
    },

    'QxPreview.Generate':{
        'en' : 'Generate',
        'ru' : 'Генерировать',
    },

    'QxPreview.Image_index' : {
        'en' : 'Image index',
        'ru' : 'Индекс изображения',
    },

    'QxPreview.Patch_mode' : {
        'en' : 'Patch mode',
        'ru' : 'Режим патча',
    },

    'QxPreview.Sample_count' : {
        'en' : 'Sample count',
        'ru' : 'Кол-во семплов',
    },
    
    'QxPreview.Fix_borders' : {
        'en' : 'Fix borders',
        'ru' : 'Фикс границ',
    },
    
    'QxPreview.Target_mask' : {
        'en' : 'Target mask',
        'ru' : 'Целевая маска',
    },

    'QxPreview.Predicted_mask' : {
        'en' : 'Predicted mask',
        'ru' : 'Предсказанная маска',
    },


    'Metric.Error':{
        'en' : 'Error',
        'ru' : 'Ошибка',
    },

    'Metric.Accuracy':{
        'en' : 'Accuracy',
        'ru' : 'Точность',
    },

    'Metric.Iteration_time' :{
        'en' : 'Iteration time',
        'ru' : 'Время итерации',
    },

    'QxMaskEditor.Thumbnail_size':{
        'en' : 'Thumbnail size',
        'ru' : 'Размер эскиза',
    },

    'QxMaskEditor.No_mask_selected':{
        'en' : 'No mask selected',
        'ru' : 'Не выбрана маска',
    },

    'QxMaskEditor.No_image_selected':{
        'en' : 'No image selected',
        'ru' : 'Не выбрано изображение',
    },

    'QxMaskEditor.Mask_type':{
        'en' : 'Mask type',
        'ru' : 'Тип маски',
    },

    'QxMaskEditor.Mask_name':{
        'en' : 'Mask name',
        'ru' : 'Имя маски',
    },

    'QxMaskEditor.Sort_by':{
        'en' : 'Sort by',
        'ru' : 'Сортировать по',
    },

    'QxMaskEditor._SortBy.Name':{
        'en' : 'Name',
        'ru' : 'Имя',
    },

    'QxMaskEditor._SortBy.PerceptualDissimilarity':{
        'en' : 'Perceptual Dissimilarity',
        'ru' : 'Зрительной непохожести',
    },

    'QxMaskEditor.Keep_view':{
        'en' : 'Keep view',
        'ru' : 'Сохранять вид',
    },

    'QxMaskEditor.Save_prev_img_mask':{
        'en' : 'Save + Previous image with mask',
        'ru' : 'Сохранить + Предыдущее изображение с маской',
    },

    'QxMaskEditor.Save_prev_img':{
        'en' : 'Save + Previous image',
        'ru' : 'Сохранить + Предыдущее изображение',
    },

    'QxMaskEditor.Copy_image':{
        'en' : 'Copy image',
        'ru' : 'Скопировать маску',
    },

    'QxMaskEditor.Copy_mask':{
        'en' : 'Copy mask',
        'ru' : 'Скопировать маску',
    },

    'QxMaskEditor.Paste_mask':{
        'en' : 'Paste mask',
        'ru' : 'Вставить маску',
    },

    'QxMaskEditor.Force_save_mask':{
        'en' : 'Force save mask',
        'ru' : 'Принудительно сохранить маску',
    },

    'QxMaskEditor.Delete_mask':{
        'en' : 'Delete mask',
        'ru' : 'Удалить маску',
    },

    'QxMaskEditor.Save_next_img':{
        'en' : 'Save + Next image',
        'ru' : 'Сохранить + Следующее изображение',
    },

    'QxMaskEditor.Save_next_img_with_mask':{
        'en' : 'Save + Next image with mask',
        'ru' : 'Сохранить + Следующее изображение с маской',
    },


    'QxMaskEditorCanvas.BW_mode':{
        'en' : 'B/W mode',
        'ru' : 'Ч/Б режим',
    },
    'QxMaskEditorCanvas.Red_overlay':{
        'en' : 'Red overlay',
        'ru' : 'Красный оверлей',
    },

    'QxMaskEditorCanvas.Green_overlay':{
        'en' : 'Green overlay',
        'ru' : 'Зелёный оверлей',
    },

    'QxMaskEditorCanvas.Blue_overlay':{
        'en' : 'Blue overlay',
        'ru' : 'Синий оверлей',
    },

    'QxMaskEditorCanvas.Opacity':{
        'en' : 'Opacity',
        'ru' : 'Прозрачность',
    },

    'QxMaskEditorCanvas.Add_delete_points':{
        'en' : 'Add/delete points',
        'ru' : 'Добавить/удалить точки',
    },

    'QxMaskEditorCanvas.Undo_action':{
        'en' : 'Undo action',
        'ru' : 'Отменить действие',
    },

    'QxMaskEditorCanvas.Redo_action':{
        'en' : 'Redo action',
        'ru' : 'Повторить действие',
    },

    'QxMaskEditorCanvas.Apply_fill_poly':{
        'en' : 'Apply fill poly',
        'ru' : 'Применить заливку полигона',
    },

    'QxMaskEditorCanvas.Apply_cut_poly':{
        'en' : 'Apply cut poly',
        'ru' : 'Применить вырезку полигоном',
    },

    'QxMaskEditorCanvas.Delete_poly':{
        'en' : 'Delete poly',
        'ru' : 'Удалить полигон',
    },

    'QxMaskEditorCanvas.Center_at_cursor':{
        'en' : 'Center at the cursor',
        'ru' : 'Центрировать на курсоре',
    },










}

pat = re.compile('@\([^\)]*\)|[^@$]+')
def L(s : str|None, lang) -> str|None:
    """
    Localize string.

    @(id)...@(id2)...
    """
    result = []
    for sub in pat.findall(s):
        if sub[:2] == '@(' and sub[-1] == ')':
            if (l := _id_to_string_dict.get(sub[2:-1], None) ) is not None:
                if (s := l.get(lang, None)) is None:
                    s = l['en']

                result.append(s)
            else:
                result.append(sub)
                print(f'No localization found for: {sub}')
        else:
            result.append(sub)

    return ''.join(result)