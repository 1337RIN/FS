document.addEventListener('DOMContentLoaded', function() {
    // Безопасное получение элементов
    const elements = {
        btnAbout: document.getElementById('btnAbout'),
        btnFeatures: document.getElementById('btnFeatures'),
        logoLink: document.getElementById('logoLink'),
        fileInput: document.getElementById('fileInput'),
        uploadForm: document.getElementById('uploadForm'),
        welcomeSection: document.getElementById('welcome'),
        preview: document.getElementById('preview'),
        result: document.getElementById('result'),
        segmentation: document.getElementById('segmentation')
    };

    // Проверяем и инициализируем только существующие элементы
    if (elements.btnAbout && elements.btnFeatures) {
        // Обработчики кнопок навигации
        elements.btnAbout.addEventListener('click', function() {
            toggleSection('about');
        });

        elements.btnFeatures.addEventListener('click', function() {
            toggleSection('features');
        });
    }

    // Обработчик логотипа
    if (elements.logoLink) {
        elements.logoLink.addEventListener('click', function(e) {
            e.preventDefault();
            this.style.transform = 'scale(0.95)';
            document.body.style.cursor = 'wait';
            setTimeout(() => {
                window.location.href = '/';
            }, 200);
        });
    }

    // Превью изображения
    if (elements.fileInput && elements.preview) {
        elements.fileInput.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {
                elements.preview.src = e.target.result;
                elements.preview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        });
    }

    // Обработчик отправки формы
    if (elements.uploadForm && elements.result && elements.segmentation) {
        elements.uploadForm.addEventListener('submit', async function(event) {
            event.preventDefault();
            const file = elements.fileInput.files[0];
            if (!file) {
                elements.result.innerText = 'Пожалуйста, выберите изображение.';
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            try {
                elements.result.innerText = 'Обработка...';
                elements.segmentation.style.display = 'none';
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                if (data.status === 'success') {
                    elements.result.innerText = `Диагностировано: ${data.disease} (Вероятность: ${(data.confidence * 100).toFixed(2)}%)`;
                    elements.segmentation.src = data.segmentation_image;
                    elements.segmentation.style.display = 'block';
                } else {
                    elements.result.innerText = `Ошибка: ${data.error}`;
                }
            } catch (error) {
                elements.result.innerText = 'Не удалось обработать изображение. Попробуйте снова.';
                console.error('Error:', error);
            }
        });
    }

    // Функция переключения секций
    function toggleSection(sectionId) {
        if (!elements.welcomeSection) return;
        
        if (!elements.welcomeSection.classList.contains('hidden')) {
            elements.welcomeSection.classList.add('hidden');
        }
        
        document.querySelectorAll('.animated-section').forEach(sec => {
            if (sec.classList) {
                sec.classList.add('hidden');
            }
        });
        
        const targetSection = document.getElementById(sectionId);
        if (targetSection && targetSection.classList) {
            targetSection.classList.remove('hidden');
        }
    }
});