import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import { Chart, BarController, BarElement, LinearScale, CategoryScale, Tooltip, Legend } from 'chart.js';
import './App.css';

Chart.register(BarController, BarElement, LinearScale, CategoryScale, Tooltip, Legend);

const API_URL = 'http://localhost:8000/api';

const DIAGNOSIS_TRANSLATIONS = {
  'MildDemented': 'Легкая степень',
  'ModerateDemented': 'Умеренная степень',
  'NonDemented': 'Норма',
  'VeryMildDemented': 'Очень легкая степень',
};

// Генерация ключа для кэша на основе файла
const generateCacheKey = (file) => {
  return `mri_analysis_${file.name}_${file.size}`;
};

function App() {
  const [image, setImage] = useState(null);
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [gradcamOpacity, setGradcamOpacity] = useState(0.5);
  const [zoomedImage, setZoomedImage] = useState(null);
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const [showExportModal, setShowExportModal] = useState(false);
  const [exportData, setExportData] = useState({
    patient_name: '',
    patient_id: '',
    patient_birth_date: '',
    patient_sex: '',
    study_date: '',
    study_description: '',
    referring_physician_name: ''
  });

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'image/*': ['.jpg', '.jpeg'],
      'application/dicom': ['.dcm']
    },
    maxFiles: 1,
    onDrop: acceptedFiles => {
      const file = acceptedFiles[0];
      if (file.name.toLowerCase().endsWith('.dcm')) {
        importFromDicom(acceptedFiles);
      } else {
        setFile(file);
        const reader = new FileReader();
        reader.onload = () => setImage(reader.result);
        reader.readAsDataURL(file);
        setResults(null);
      }
    }
  });

  const handleImageClick = (imageSrc) => {
    setZoomedImage(imageSrc);
  };

  const closeZoomedImage = () => {
    setZoomedImage(null);
  };

  const analyzeImage = async () => {
    if (!file) return;
    
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post(`${API_URL}/analyze`, formData);
      setResults(response.data);
    } catch (error) {
      console.error('Analysis error:', error);
      alert('Ошибка при анализе изображения');
    } finally {
      setLoading(false);
    }
  };

  const classifyImage = async () => {
    if (!file) return;
    
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post(`${API_URL}/classify`, formData);
      setResults({
        classification: response.data,
        interpretation: null
      });
    } catch (error) {
      console.error('Classification error:', error);
      alert('Ошибка при классификации изображения');
    } finally {
      setLoading(false);
    }
  };

  const handleExportDataChange = (e) => {
    const { name, value } = e.target;
    setExportData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const exportToDicom = async () => {
    if (!file) return;
    
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('export_data', JSON.stringify(exportData));
      
      const response = await axios.post(`${API_URL}/export/dicom`, formData, {
        responseType: 'blob',
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      // Создаем ссылку для скачивания
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'mri_export.dcm');
      document.body.appendChild(link);
      link.click();
      link.remove();
      
      setShowExportModal(false);
    } catch (error) {
      console.error('Export error:', error);
      if (error.response?.status === 422) {
        alert('Ошибка в данных формы. Пожалуйста, проверьте все поля.');
      } else {
        alert('Ошибка при экспорте в DICOM');
      }
    } finally {
      setLoading(false);
    }
  };

  const importFromDicom = async (acceptedFiles) => {
    const dicomFile = acceptedFiles[0];
    if (!dicomFile) return;
    
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', dicomFile);
      formData.append('output_format', 'image');
      
      const response = await axios.post(`${API_URL}/import/dicom`, formData, {
        responseType: 'blob'
      });
      
      // Создаем Blob из ответа
      const blob = new Blob([response.data], { type: 'image/jpeg' });
      
      // Создаем URL для изображения
      const imageUrl = URL.createObjectURL(blob);
      setImage(imageUrl);
      
      // Создаем File объект из Blob
      const file = new File([blob], 'imported_image.jpg', { 
        type: 'image/jpeg',
        lastModified: new Date().getTime()
      });
      setFile(file);
      
      // Запускаем классификацию с тем же файлом
      await classifyImage();
      
    } catch (error) {
      console.error('Import error:', error);
      alert('Ошибка при импорте DICOM файла');
    } finally {
      setLoading(false);
    }
  };

  const renderDiagnosisChart = (probabilities) => {
    if (!probabilities || !chartRef.current) return;

    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    const chartCtx = chartRef.current.getContext('2d');
    if (!chartCtx) return;
    
    const labels = Object.keys(DIAGNOSIS_TRANSLATIONS);
    const data = labels.map(label => probabilities[label]);
    
    chartInstance.current = new Chart(chartCtx, {
      type: 'bar',
      data: {
        labels: labels.map(key => DIAGNOSIS_TRANSLATIONS[key]),
        datasets: [{
          data: data,
          backgroundColor: [
            'rgba(255, 99, 132, 0.7)',
            'rgba(54, 162, 235, 0.7)',
            'rgba(75, 192, 192, 0.7)',
            'rgba(255, 206, 86, 0.7)'
          ],
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            max: 1,
            ticks: {
              callback: value => `${(value * 100).toFixed(0)}%`
            }
          }
        },
        plugins: {
          legend: {
            display: false
          }
        }
      }
    });
  };

  useEffect(() => {
    if (results?.classification?.probabilities) {
      renderDiagnosisChart(results.classification.probabilities);
    }
  }, [results]);

  useEffect(() => {
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, []);

  return (
    <div className="app-container">
      <div className="upload-section">
        <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
          <input {...getInputProps()} />
          <p>{image ? file.name : 'Перетащите МРТ-снимок или DICOM файл сюда'}</p>
        </div>
        <div className="button-group">
          <button 
            onClick={analyzeImage} 
            disabled={!image || loading}
            className="analyze-btn"
          >
            {loading ? 'Анализ...' : 'Полный анализ'}
          </button>
          <button 
            onClick={classifyImage} 
            disabled={!image || loading}
            className="classify-btn"
          >
            {loading ? 'Классификация...' : 'Классифицировать'}
          </button>
          <button 
            onClick={() => setShowExportModal(true)} 
            disabled={!image || loading}
            className="export-btn"
          >
            Экспорт в DICOM
          </button>
        </div>
      </div>

      {/* Модальное окно для экспорта */}
      {showExportModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h2>Экспорт в DICOM</h2>
            <form onSubmit={(e) => { e.preventDefault(); exportToDicom(); }}>
              <div className="form-group">
                <label>Имя пациента *</label>
                <input
                  type="text"
                  name="patient_name"
                  value={exportData.patient_name}
                  onChange={handleExportDataChange}
                  required
                />
              </div>
              <div className="form-group">
                <label>ID пациента</label>
                <input
                  type="text"
                  name="patient_id"
                  value={exportData.patient_id}
                  onChange={handleExportDataChange}
                />
              </div>
              <div className="form-group">
                <label>Дата рождения</label>
                <input
                  type="date"
                  name="patient_birth_date"
                  value={exportData.patient_birth_date}
                  onChange={handleExportDataChange}
                />
              </div>
              <div className="form-group">
                <label>Пол</label>
                <select
                  name="patient_sex"
                  value={exportData.patient_sex}
                  onChange={handleExportDataChange}
                >
                  <option value="">Выберите пол</option>
                  <option value="M">Мужской</option>
                  <option value="F">Женский</option>
                  <option value="O">Другой</option>
                </select>
              </div>
              <div className="form-group">
                <label>Дата исследования</label>
                <input
                  type="date"
                  name="study_date"
                  value={exportData.study_date}
                  onChange={handleExportDataChange}
                />
              </div>
              <div className="form-group">
                <label>Описание исследования</label>
                <input
                  type="text"
                  name="study_description"
                  value={exportData.study_description}
                  onChange={handleExportDataChange}
                />
              </div>
              <div className="form-group">
                <label>Имя направляющего врача</label>
                <input
                  type="text"
                  name="referring_physician_name"
                  value={exportData.referring_physician_name}
                  onChange={handleExportDataChange}
                />
              </div>
              <div className="modal-buttons">
                <button type="submit" disabled={loading}>
                  {loading ? 'Экспорт...' : 'Экспортировать'}
                </button>
                <button type="button" onClick={() => setShowExportModal(false)}>
                  Отмена
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {results && (
        <div className="results-row">
          {results.interpretation && (
            <>
              {/* Блок Grad-CAM с оригинальным изображением */}
              <div className="gradcam-block">
                <div className="gradcam-container">
                  <div className="image-wrapper" onClick={() => handleImageClick(image)}>
                    <img 
                      src={image} 
                      alt="Оригинальное изображение" 
                      className="original-underlay"
                      style={{ width: '224px', height: '224px', objectFit: 'cover' }}
                    />
                    <img 
                      src={`data:image/jpeg;base64,${results.interpretation.additional_info.heatmap_img}`} 
                      alt="Grad-CAM" 
                      className="heatmap-overlay"
                      style={{ 
                        opacity: gradcamOpacity,
                        width: '224px',
                        height: '224px',
                        objectFit: 'cover',
                        position: 'absolute',
                        top: 0,
                        left: 0
                      }}
                    />
                  </div>
                </div>
                
                <div className="opacity-control">
                  <label>Прозрачность Grad-CAM:</label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={gradcamOpacity}
                    onChange={(e) => setGradcamOpacity(e.target.value)}
                  />
                </div>
              </div>

              {/* Блок LIME */}
              {results.interpretation.additional_info.lime_img && (
                <div className="lime-block">
                  <div className="lime-container">
                    <div className="image-wrapper" onClick={() => handleImageClick(`data:image/jpeg;base64,${results.interpretation.additional_info.lime_img}`)}>
                      <img 
                        src={`data:image/jpeg;base64,${results.interpretation.additional_info.lime_img}`} 
                        alt="LIME объяснение"
                        style={{ width: '224px', height: '224px', objectFit: 'cover' }}
                      />
                    </div>
                  </div>
                </div>
              )}
            </>
          )}

          {/* Блок графика и диагноза */}
          <div className="chart-block">
            <div className="chart-container">
              <canvas ref={chartRef} />
            </div>
            <div className="diagnosis-box">
              <h3>Результат:</h3>
              <p className="diagnosis">
                {DIAGNOSIS_TRANSLATIONS[results.classification.class_name] || results.classification.class_name}
              </p>
              <p className="confidence">
                Уверенность: {(results.classification.confidence * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Модальное окно для увеличенного изображения */}
      {zoomedImage && (
        <div className="zoom-modal" onClick={closeZoomedImage}>
          <div className="zoom-content">
            <img src={zoomedImage} alt="Увеличенное изображение" />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;