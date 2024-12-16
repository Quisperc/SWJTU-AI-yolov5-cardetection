// script.js

// 获取文件选择元素
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');

// 监听文件选择变化事件
fileInput.addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const fileReader = new FileReader();

        // 如果是图片文件
        if (file.type.startsWith('image/')) {
            fileReader.onload = function(e) {
                // 创建一个 img 元素来显示图片
                const img = document.createElement('img');
                img.src = e.target.result;
                img.alt = 'Uploaded Image Preview';
                img.classList.add('preview-image');

                // 清除之前的预览并添加新预览
                previewContainer.innerHTML = '';
                previewContainer.innerHTML = '<h4>图片预览</h4>';
                previewContainer.appendChild(img);
            };
            fileReader.readAsDataURL(file);
        }

        // 如果是视频文件
        else if (file.type.startsWith('video/')) {
            fileReader.onload = function(e) {
                // 创建一个 video 元素来显示视频
                const video = document.createElement('video');
                video.src = e.target.result;
                video.controls = true;
                video.autoplay = true;
                video.classList.add('preview-video');

                // 清除之前的预览并添加新预览
                previewContainer.innerHTML = '';
                previewContainer.innerHTML = '<h4>视频预览</h4>';
                previewContainer.appendChild(video);
            };
            fileReader.readAsDataURL(file);
        }
    }
});
