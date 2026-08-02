

# 🐟 Sistema de Detección y Clasificación de Peces

Un sistema completo de visión por computadora para detectar y clasificar especies de peces utilizando un enfoque de flujo de trabajo (pipeline) de dos etapas. Este proyecto combina YOLOv8 para la detección de peces con MobileNetV2 para la clasificación de especies, optimizado para despliegue en dispositivos móviles.

![Model Architecture](https://image2url.com/images/1758981625298-2d6ce34f-00b1-4739-b7fc-e4d743661107.jpg)

## 📋 Tabla de Contenidos

- [Descripción General](#overview)
- [Arquitectura](#architecture)
- [Características](#features)
- [Instalación](#installation)
- [Uso](#usage)
- [Conversión de Modelos para Android](#model-conversion-for-android)
- [Integración en Android](#android-integration)
- [Rendimiento](#performance)
- [Conjunto de Datos](#dataset)
- [Contribuir](#contributing)
- [Licencia](#license)

## 🎯 Descripción General

Este proyecto implementa un sofisticado sistema de detección y clasificación de peces utilizando un flujo de trabajo de dos etapas:

1. **Etapa de Detección**: El modelo YOLOv8n detecta peces y crea cajas delimitadoras
2. **Etapa de Clasificación**: El modelo MobileNetV2 clasifica los peces detectados en 18 especies

El sistema está diseñado para aplicaciones en tiempo real y despliegue en dispositivos móviles, lo que lo hace ideal para aplicaciones Android, investigación marina y monitoreo acuícola.

## 🏗️ Arquitectura

### Flujo de Trabajo de Dos Etapas

```
Input Image → YOLOv8 Detection → Fish Cropping → MobileNetV2 Classification → Species Output
```

### Detalles del Modelo

| Modelo | Propósito | Tamaño de Entrada | Clases | Rendimiento |
|-------|---------|------------|---------|-------------|
| **YOLOv8n** | Detección de Peces | 640×640 | 1 (pez) | 99.97% Precisión, 100% Recall |
| **MobileNetV2** | Clasificación de Especies | 224×224 | 18 especies | 99.50% mAP@0.5 |

### Especies de Peces Soportadas

El modelo de clasificación puede identificar 18 especies diferentes de peces:

- Bass GT Sea, Bass Sea
- Black GT Sea Sprat, Black Sea Sprat  
- Bream GT Gilt-Head, Bream GT Red Sea, Bream Gilt-Head, Bream Red Sea
- GT Hourse Mackerel, Hourse Mackerel
- GT Mullet Red, GT Mullet Red Striped, Mullet Red, Mullet Red Striped
- GT Shrimp, Shrimp
- GT Trout, Trout

## ✨ Características

- **Detección en Tiempo Real**: Detección rápida de peces utilizando YOLOv8n
- **Alta Precisión**: 99.97% de precisión en la detección de peces
- **Clasificación de Especies**: 18 especies de peces con 99.50% de mAP
- **Optimizado para Móviles**: Modelos convertidos a TensorFlow Lite para Android
- **Uso Flexible**: Utiliza solo detección, solo clasificación, o el flujo completo
- **Procesamiento por Lotes**: Procesa imágenes, videos y carpetas en lote
- **Soporte para Webcam**: Procesamiento en tiempo real con webcam
- **Fácil Integración**: API simple para integración en aplicaciones Android

## 🚀 Instalación

### Prerequisitos

- Python 3.8+
- GPU compatible con CUDA (recomendado)
- Android Studio (para despliegue en móviles)

### Instalar Dependencias

```bash
# Clone the repository
git clone https://github.com/yourusername/fish-detection-classification.git
cd fish-detection-classification

# Install Python dependencies
pip install -r requirements.txt
```

### Requisitos

```
ultralytics>=8.0.0
tensorflow>=2.10.0
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.5.0
Pillow>=8.0.0
```

## 📖 Uso

### 1. Solo Detección (YOLOv8)

```bash
# Detect fish in an image
python Yolov8n_detection/inference.py --image fish.jpg

# Detect and crop fish regions
python Yolov8n_detection/inference.py --image fish.jpg --crop

# Process video
python Yolov8n_detection/inference.py --video fish_video.mp4 --output result.mp4

# Real-time webcam detection
python Yolov8n_detection/inference.py --webcam
```

### 2. Solo Clasificación (MobileNetV2)

```bash
# Classify a single fish image
python mobilenetV3_classfication/inference.py --image cropped_fish.jpg

# Batch classify multiple images
python mobilenetV3_classfication/inference.py --batch cropped_fish_folder/

# Real-time webcam classification
python mobilenetV3_classfication/inference.py --webcam
```

### 3. Flujo Completo (Pipeline)

```bash
# Full pipeline: detect + classify
python fish_pipeline.py --image fish.jpg

# Process video with full pipeline
python fish_pipeline.py --video fish_video.mp4 --output result.mp4

# Custom confidence thresholds
python fish_pipeline.py --image fish.jpg --detection-conf 0.6 --classification-conf 0.4
```

## 📱 Conversión de Modelos para Android

### Paso 1: Convertir YOLOv8 a TensorFlow Lite

```python
# Convert YOLOv8 to TensorFlow Lite
from ultralytics import YOLO

# Load trained model
model = YOLO('Yolov8n_detection/best.pt')

# Export to TensorFlow Lite
model.export(format='tflite', imgsz=640, int8=True, optimize=True)
```

### Paso 2: Convertir MobileNetV2 a TensorFlow Lite

```python
# Convert MobileNetV2 to TensorFlow Lite
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('mobilenetV3_classfication/best_model.h5')

# Convert to TensorFlow Lite with INT8 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable INT8 quantization for better performance
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]

# Convert model
tflite_model = converter.convert()

# Save model
with open('fish_classifier_int8.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Paso 3: Optimizar para Móviles

```python
# Additional optimization for mobile deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable all optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Enable INT8 quantization
converter.target_spec.supported_types = [tf.int8]

# Enable GPU acceleration (if available)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# Convert and save
tflite_model = converter.convert()
```

## 📱 Integración en Android

### Usar CameraX para Procesamiento en Tiempo Real

#### 1. Agregar Dependencias a `build.gradle`

```gradle
dependencies {
    implementation 'androidx.camera:camera-core:1.3.0'
    implementation 'androidx.camera:camera-camera2:1.3.0'
    implementation 'androidx.camera:camera-lifecycle:1.3.0'
    implementation 'androidx.camera:camera-view:1.3.0'
    
    // TensorFlow Lite
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
}
```

#### 2. Implementación de CameraX

```kotlin
class FishDetectionActivity : AppCompatActivity() {
    private lateinit var cameraProvider: ProcessCameraProvider
    private lateinit var imageAnalyzer: ImageAnalysis
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_fish_detection)
        
        setupCamera()
    }
    
    private fun setupCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun bindCameraUseCases() {
        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(Size(640, 640))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            
        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this)) { imageProxy ->
            processImage(imageProxy)
        }
        
        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
        
        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this, cameraSelector, imageAnalysis
            )
        } catch (e: Exception) {
            Log.e("CameraX", "Use case binding failed", e)
        }
    }
    
    private fun processImage(imageProxy: ImageProxy) {
        // Convert ImageProxy to Bitmap
        val bitmap = imageProxyToBitmap(imageProxy)
        
        // Run fish detection
        val detections = runFishDetection(bitmap)
        
        // For each detection, run classification
        detections.forEach { detection ->
            val croppedFish = cropFish(bitmap, detection.bbox)
            val species = runFishClassification(croppedFish)
            
            // Update UI with results
            updateUI(detection, species)
        }
        
        imageProxy.close()
    }
}
```

#### 3. Integración con TensorFlow Lite

```kotlin
class FishDetectionModel {
    private var detectionInterpreter: Interpreter? = null
    private var classificationInterpreter: Interpreter? = null
    
    fun loadModels() {
        // Load detection model
        val detectionOptions = Interpreter.Options().apply {
            setNumThreads(4)
            setUseNNAPI(true)
        }
        detectionInterpreter = Interpreter(loadModelFile("fish_detector_int8.tflite"), detectionOptions)
        
        // Load classification model
        val classificationOptions = Interpreter.Options().apply {
            setNumThreads(4)
            setUseNNAPI(true)
        }
        classificationInterpreter = Interpreter(loadModelFile("fish_classifier_int8.tflite"), classificationOptions)
    }
    
    fun detectFish(bitmap: Bitmap): List<Detection> {
        val input = preprocessImage(bitmap, 640, 640)
        val output = Array(1) { Array(25200) { FloatArray(85) } }
        
        detectionInterpreter?.run(input, output)
        
        return postprocessDetections(output[0])
    }
    
    fun classifyFish(bitmap: Bitmap): String {
        val input = preprocessImage(bitmap, 224, 224)
        val output = Array(1) { FloatArray(18) }
        
        classificationInterpreter?.run(input, output)
        
        return getSpeciesName(output[0])
    }
}
```

#### 4. Permisos Requeridos

```xml
<!-- AndroidManifest.xml -->
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-feature android:name="android.hardware.camera" android:required="true" />
<uses-feature android:name="android.hardware.camera.autofocus" android:required="false" />
```

### Pasos de Integración

1. **Agregar Modelos**: Colocar los archivos `.tflite` en `app/src/main/assets/`
2. **Implementar CameraX**: Usar el código proporcionado para la integración de la cámara
3. **Agregar TensorFlow Lite**: Incluir el código de inferencia del modelo
4. **Actualizar UI**: Mostrar los resultados de detección y clasificación
5. **Probar Rendimiento**: Optimizar para tu dispositivo objetivo

## 📊 Rendimiento

### Rendimiento del Modelo

| Métrica | Detección YOLOv8 | Clasificación MobileNetV2 |
|--------|------------------|----------------------------|
| **Precisión** | 99.97% | 99.50% |
| **Recall** | 100.00% | 99.50% |
| **mAP@0.5** | 99.50% | 99.50% |
| **Tiempo de Inferencia** | ~13ms | ~8ms |
| **Tamaño del Modelo** | 6.2MB | 4.1MB |

### Rendimiento en Móviles (Cuantizado INT8)

| Dispositivo | Tiempo de Detección | Tiempo de Clasificación | Flujo Total |
|--------|----------------|-------------------|----------------|
| **Pixel 6** | 25ms | 15ms | 40ms |
| **Samsung S21** | 30ms | 18ms | 48ms |
| **OnePlus 9** | 28ms | 16ms | 44ms |

## 📚 Conjunto de Datos (Dataset)

### Conjunto de Detección (YOLOv8)
- **Clases**: 7 especies de peces (Pomfret, Mackerel, Black Snapper, Indian Carp, Prawn, Pink Perch, Black Pomfret)
- **Formato**: Formato YOLO con divisiones train/val/test
- **Resolución**: 640×640 píxeles
- **Enlace al Dataset**: [Enlace por agregar]

### Conjunto de Clasificación (MobileNetV2)
- **Clases**: 18 especies de peces (Bass, Bream, Mullet, Shrimp, Trout, etc.)
- **Formato**: Clasificación de imágenes con divisiones train/val/test
- **Resolución**: 224×224 píxeles
- **Enlace al Dataset**: [Enlace por agregar]

### Citación

```bibtex
@dataset{fish_detection_classification_2024,
  title={Fish Detection and Classification Dataset},
  author={Dataset Provider Name},
  year={2024},
  url={https://github.com/dataset-provider/fish-datasets}
}
```

### Agradecimientos

Agradecemos a los proveedores del conjunto de datos por su valiosa contribución a la investigación marina y a las aplicaciones de visión por computadora.

## 🚧 Estado de Desarrollo

> **"La innovación distingue entre un líder y un seguidor."** - Steve Jobs

Este proyecto se encuentra actualmente en **fase de desarrollo activo y pruebas**. Estamos trabajando continuamente para mejorar la precisión, el rendimiento y la experiencia de usuario de nuestro sistema de detección y clasificación de peces.

### 🔄 Qué Viene Después

- **Mejora de la Precisión de Detección**: Agregar más imágenes negativas a los datos de entrenamiento para una mejor clasificación ambiental
- **Optimización para Móviles**: Optimización adicional para despliegue en Android y rendimiento en tiempo real
- **Soporte Extendido de Especies**: Agregar más especies de peces a nuestro modelo de clasificación
- **Documentación**: Expandir tutoriales y guías de integración

### 🧪 Fase Actual de Pruebas

Estamos probando y refinando activamente:
- Rendimiento del modelo en diferentes dispositivos y entornos
- Precisión en condiciones reales bajo diversas iluminaciones
- Optimización del despliegue en dispositivos móviles

### 💡 Contribuir al Desarrollo

¡Agradecemos las contribuciones de la comunidad! Ya sea que te intereses por:
- **Mejoras del Modelo**: Mejorar la precisión de detección y clasificación
- **Desarrollo Móvil**: Desarrollo y optimización de aplicaciones Android
- **Pruebas**: Ayudarnos a probar en diferentes dispositivos y escenarios

Tus contribuciones nos ayudan a construir un sistema de detección de peces mejor y más robusto para la comunidad de investigación marina.

## 🤝 Contribuir

¡Bienvenidas las contribuciones! Por favor consulta nuestras [Guías de Contribución](CONTRIBUTING.md) para más detalles.

### Cómo Contribuir

1. Haz un fork del repositorio
2. Crea una rama de funcionalidad (`git checkout -b feature/NuevaFuncionalidad`)
3. Realiza commit de tus cambios (`git commit -m 'Agregar NuevaFuncionalidad'`)
4. Haz push a la rama (`git push origin feature/NuevaFuncionalidad`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - consulta el archivo [LICENSE](LICENSE) para más detalles.

## 👥 Contribuyentes

- **[Aditya Sharma](https://github.com/Aditya-54)** 
- **[Pranshul Gupta](https://github.com/pranshulgupta33940)** 
- **[Pradhuman singh rajvi](https://github.com/techhuman22)**
- **[Nishant Chaudhary](https://github.com/hero0p)**
- **[Sameer Kaushik](https://github.com/Sameer060405)**

## 🙏 Agradecimientos

- [Ultralytics](https://github.com/ultralytics/ultralytics) por la implementación de YOLOv8
- [TensorFlow](https://www.tensorflow.org/) por MobileNetV2 y TensorFlow Lite
- [CameraX](https://developer.android.com/training/camerax) por la integración de cámara en Android
- Contribuyentes del conjunto de datos de peces e investigadores de biología marina

## 📞 Contacto

- **Enlace del Proyecto**: [https://github.com/yourusername/fish-detection-classification](https://github.com/yourusername/fish-detection-classification)
- **Problemas (Issues)**: [GitHub Issues](https://github.com/yourusername/fish-detection-classification/issues)
- **Discusiones**: [GitHub Discussions](https://github.com/yourusername/fish-detection-classification/discussions)

---

⭐ **¡Marca con una estrella este repositorio** si te resultó útil!

🐟 **¡Feliz Detección de Peces!** 🐟
