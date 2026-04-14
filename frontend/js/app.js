// Main Application Logic

class BiasDetectionApp {
  constructor() {
    this.sessionId = null;
    this.columns = [];
    this.uniques = {};
    this.analysisResult = null;
    this.metrics = null;
    this.predictionData = null; // Store prediction dataset CSV text
    this.predictionColumns = [];
    this.lastAnalyzeResponse = null;
    this.lastAnalysisConfig = null;
    this.modelComparisonMetrics = [
      'accuracy',
      'selection_rate_difference',
      'equal_opportunity_difference',
      'average_odds_difference',
      'false_positive_rate_difference',
    ];
    this.init();
  }

  async init() {
    // Bind event listeners
    document.getElementById('upload-btn').addEventListener('click', () => this.uploadDataset());
    document.getElementById('upload-prediction-btn').addEventListener('click', () => this.uploadPredictionDataset());
    document.getElementById('analyze-btn').addEventListener('click', () => this.runAnalysis());
    document.getElementById('mitigate-btn').addEventListener('click', () => this.runMitigation());
    document.getElementById('clear-btn').addEventListener('click', () => this.clearResults());

    // Bind change listeners for dropdowns
    document.getElementById('protected-attr').addEventListener('change', () => this.onProtectedAttrChange());
    document.getElementById('priv-val').addEventListener('change', () => this.onPrivilegedValueChange());
    document.getElementById('detection-type').addEventListener('change', () => this.onDetectionTypeChange());
    document.getElementById('model-input-mode').addEventListener('change', () => this.onModelInputModeChange());
    

    // Bind file input change listener for prediction dataset
    document.getElementById('prediction-file').addEventListener('change', () => this.onPredictionFileSelected());

    // Display current API URL and handle change
    const apiUrl = API.getBaseUrl();
    document.getElementById('api-url').textContent = apiUrl;
    document.getElementById('change-api-btn').addEventListener('click', () => {
      const newUrl = prompt('Enter Backend API URL (e.g. https://your-tunnel.ngrok.io):', apiUrl);
      if (newUrl) API.setBaseUrl(newUrl);
    });

    // Show warning if on HTTPS but API is HTTP
    if (window.location.protocol === 'https:' && apiUrl.startsWith('http://')) {
      document.getElementById('mixed-content-warning').classList.remove('hidden');
    }

    // Load metrics from backend
    await this.loadMetrics();
    this.onDetectionTypeChange();
  }

  async loadMetrics() {
    try {
      const response = await API.getMetrics();
      if (typeof response === 'string') {
        throw new Error('Server returned HTML instead of JSON. Check your API URL and ngrok status.');
      }
      this.metrics = response;
      console.log('Metrics loaded:', this.metrics);
    } catch (error) {
      console.error('Failed to load metrics:', error);
      Utils.showError('Failed to load metrics from server: ' + (error.message || error));
    }
  }

  async uploadDataset() {
    const fileInput = document.getElementById('dataset-file');
    const file = fileInput.files[0];

    if (!file) {
      Utils.showError('Please select a CSV file');
      return;
    }

    try {
      this.setLoading(true);
      const response = await API.uploadDataset(file);

      this.sessionId = response.session_id;
      this.columns = response.columns || [];
      this.uniques = response.uniques || {};

      // Update UI
      document.getElementById('session-id').textContent = this.sessionId;
      document.getElementById('session-info').classList.remove('hidden');
      document.getElementById('analysis-section').classList.remove('hidden');
      document.getElementById('mitigation-section').classList.remove('hidden');

      // Populate dropdowns
      this.populateDropdowns();

      // Populate metrics based on detection type
      this.populateMetrics();

      Utils.showSuccess('Dataset uploaded successfully!');
    } catch (error) {
      Utils.showError(error);
    } finally {
      this.setLoading(false);
    }
  }

  populateDropdowns() {
    const protectedAttrSelect = document.getElementById('protected-attr');
    const labelColSelect = document.getElementById('label-col');

    // Clear existing options
    protectedAttrSelect.innerHTML = '';
    labelColSelect.innerHTML = '';

    // Add options
    this.columns.forEach(col => {
      const option1 = document.createElement('option');
      option1.value = col;
      option1.textContent = col;
      protectedAttrSelect.appendChild(option1);

      const option2 = document.createElement('option');
      option2.value = col;
      option2.textContent = col;
      labelColSelect.appendChild(option2);
    });

    // Trigger initial protected attribute change to populate values
    if (this.columns.length > 0) {
      this.onProtectedAttrChange();
    }
  }

  onProtectedAttrChange() {
    const protectedAttr = document.getElementById('protected-attr').value;
    if (!protectedAttr) return;

    const values = this.uniques[protectedAttr] || [];

    // Populate privileged and unprivileged dropdowns
    const privValSelect = document.getElementById('priv-val');
    const unprivValSelect = document.getElementById('unpriv-val');

    privValSelect.innerHTML = '';
    unprivValSelect.innerHTML = '';

    values.forEach(val => {
      const option1 = document.createElement('option');
      option1.value = val;
      option1.textContent = val;
      privValSelect.appendChild(option1);

      const option2 = document.createElement('option');
      option2.value = val;
      option2.textContent = val;
      unprivValSelect.appendChild(option2);
    });

    // Auto-select different values if binary
    if (values.length === 2) {
      privValSelect.value = values[1];
      unprivValSelect.value = values[0];
    } else if (values.length > 0) {
      privValSelect.value = values[0];
      if (values.length > 1) {
        unprivValSelect.value = values[1];
      }
    }
  }

  onPrivilegedValueChange() {
    // Auto-update unprivileged value when privileged changes (for binary attributes)
    const protectedAttr = document.getElementById('protected-attr').value;
    if (!protectedAttr) return;

    const values = this.uniques[protectedAttr] || [];
    if (values.length !== 2) return; // Only auto-update for binary attributes

    const privVal = document.getElementById('priv-val').value;
    const unprivValSelect = document.getElementById('unpriv-val');

    // Select the other value
    const otherValue = values.find(v => v !== privVal);
    if (otherValue !== undefined) {
      unprivValSelect.value = otherValue;
    }
  }

  onDetectionTypeChange() {
    // Re-populate metrics when detection type changes
    this.populateMetrics();

    // Toggle model-specific controls and prediction section
    const detectionType = document.getElementById('detection-type').value;
    const predictionSection = document.getElementById('prediction-upload-section');
    const modelInputModeGroup = document.getElementById('model-input-mode-group');
    const modelTypeGroup = document.getElementById('model-type-group');
    const datasetMitigationControls = document.getElementById('dataset-mitigation-controls');
    const modelMitigationControls = document.getElementById('model-mitigation-controls');
    const suggestionBox = document.getElementById('suggestion-box');
    const repairGroup = document.getElementById('repair-level-group');
    const mitigateBtn = document.getElementById('mitigate-btn');

    if (detectionType === 'Model Bias Detection') {
      modelInputModeGroup.style.display = 'block';
      datasetMitigationControls.style.display = 'none';
      modelMitigationControls.style.display = 'block';
      mitigateBtn.textContent = 'Run Model Comparison';
      suggestionBox.classList.add('hidden');
      repairGroup.style.display = 'none';
      this.onModelInputModeChange();
    } else {
      predictionSection.classList.add('hidden');
      modelInputModeGroup.style.display = 'none';
      modelTypeGroup.style.display = 'none';
      datasetMitigationControls.style.display = 'block';
      modelMitigationControls.style.display = 'none';
      mitigateBtn.textContent = 'Apply Mitigation';
      // Clear prediction data when switching back to dataset detection
      this.predictionData = null;
      this.predictionColumns = [];
      document.getElementById('prediction-info').classList.add('hidden');
    }
  }

  onModelInputModeChange() {
    const detectionType = document.getElementById('detection-type').value;
    const mode = document.getElementById('model-input-mode').value;
    const predictionSection = document.getElementById('prediction-upload-section');
    const modelTypeGroup = document.getElementById('model-type-group');

    if (detectionType !== 'Model Bias Detection') {
      predictionSection.classList.add('hidden');
      modelTypeGroup.style.display = 'none';
      return;
    }

    if (mode === 'upload_pred_dataset') {
      predictionSection.classList.remove('hidden');
      modelTypeGroup.style.display = 'none';
    } else {
      predictionSection.classList.add('hidden');
      modelTypeGroup.style.display = 'block';
    }
  }
  

  onPredictionFileSelected() {
    // When a prediction file is selected, populate the column dropdowns
    const fileInput = document.getElementById('prediction-file');
    const file = fileInput.files[0];

    if (!file) return;

    // Read the CSV to get columns
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const lines = text.split('\n');
      if (lines.length > 0) {
        const headers = lines[0].split(',').map(h => h.trim());
        this.predictionColumns = headers;

        // Populate dropdowns
        const predLabelSelect = document.getElementById('pred-label-col');
        const predProbaSelect = document.getElementById('pred-proba-col');
        const idColSelect = document.getElementById('id-col');

        predLabelSelect.innerHTML = '';
        predProbaSelect.innerHTML = '<option value="">-- None --</option>';

        // Add original dataset columns to id-col dropdown
        idColSelect.innerHTML = '<option value="">-- None (use row order) --</option>';
        this.columns.forEach(col => {
          const option = document.createElement('option');
          option.value = col;
          option.textContent = col;
          idColSelect.appendChild(option);
        });

        headers.forEach(col => {
          const option1 = document.createElement('option');
          option1.value = col;
          option1.textContent = col;
          predLabelSelect.appendChild(option1);

          const option2 = document.createElement('option');
          option2.value = col;
          option2.textContent = col;
          predProbaSelect.appendChild(option2);
        });
      }
    };
    reader.readAsText(file);
  }

  async uploadPredictionDataset() {
    const fileInput = document.getElementById('prediction-file');
    const file = fileInput.files[0];

    if (!file) {
      Utils.showError('Please select a prediction CSV file');
      return;
    }

    try {
      this.setLoading(true);

      // Read the file as text
      const reader = new FileReader();
      reader.onload = (e) => {
        this.predictionData = e.target.result;
        document.getElementById('prediction-info').classList.remove('hidden');
        Utils.showSuccess('Prediction dataset loaded successfully!');
        this.setLoading(false);
      };
      reader.onerror = () => {
        Utils.showError('Failed to read prediction file');
        this.setLoading(false);
      };
      reader.readAsText(file);
    } catch (error) {
      Utils.showError(error);
      this.setLoading(false);
    }
  }

  populateMetrics() {
    if (!this.metrics) return;

    const detectionType = document.getElementById('detection-type').value;
    const container = document.getElementById('metrics-container');
    container.innerHTML = '';

    // Determine which metrics to show
    let metricsToShow = {};
    if (detectionType === 'Dataset Bias Detection') {
      metricsToShow = this.metrics.dataset_metrics || {};
    } else {
      metricsToShow = this.metrics.classification_metrics || {};
    }

    // Keep accuracy out of detection-panel metric choices (used only in model comparison views).
    if (detectionType === 'Model Bias Detection') {
      metricsToShow = Object.fromEntries(
        Object.entries(metricsToShow).filter(([key]) => key !== 'accuracy')
      );
    }

    // Create checkbox for each metric
    Object.entries(metricsToShow).forEach(([key, info]) => {
      const metricItem = document.createElement('div');
      metricItem.className = 'metric-item';

      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.id = `metric-${key}`;
      checkbox.value = key;
      checkbox.checked = true; // Select all by default

      const label = document.createElement('label');
      label.htmlFor = `metric-${key}`;
      label.style.cursor = 'pointer';
      label.style.display = 'block';

      const nameDiv = document.createElement('div');
      nameDiv.className = 'metric-name';
      nameDiv.textContent = info.name || key;

      const descDiv = document.createElement('div');
      descDiv.className = 'metric-description';
      descDiv.textContent = info.description || '';

      label.appendChild(checkbox);
      label.appendChild(nameDiv);
      label.appendChild(descDiv);

      metricItem.appendChild(label);

      // Toggle selected class on click
      metricItem.addEventListener('click', (e) => {
        if (e.target !== checkbox) {
          checkbox.checked = !checkbox.checked;
        }
        if (checkbox.checked) {
          metricItem.classList.add('selected');
        } else {
          metricItem.classList.remove('selected');
        }
      });

      // Initialize selected state
      if (checkbox.checked) {
        metricItem.classList.add('selected');
      }

      container.appendChild(metricItem);
    });
  }

  async runAnalysis() {
    if (!this.sessionId) {
      Utils.showError('Please upload a dataset first');
      return;
    }

    const detectionType = document.getElementById('detection-type').value;
    const protectedAttr = document.getElementById('protected-attr').value;
    const labelCol = document.getElementById('label-col').value;
    const privVal = document.getElementById('priv-val').value;
    const unprivVal = document.getElementById('unpriv-val').value;
    const modelType = document.getElementById('model-type').value;
    const modelInputMode = document.getElementById('model-input-mode').value;

    // Get selected metrics
    const selectedMetrics = [];
    document.querySelectorAll('#metrics-container input[type="checkbox"]:checked').forEach(cb => {
      selectedMetrics.push(cb.value);
    });

    if (!protectedAttr || !labelCol) {
      Utils.showError('Please select protected attribute and label column');
      return;
    }

    if (!privVal || !unprivVal) {
      Utils.showError('Please select privileged and unprivileged values');
      return;
    }

    if (selectedMetrics.length === 0) {
      Utils.showError('Please select at least one metric');
      return;
    }

    const payload = {
      session_id: this.sessionId,
      detection_type: detectionType,
      protected_attr: protectedAttr,
      label_column: labelCol,
      privileged_value: privVal,
      unprivileged_value: unprivVal,
      selected_metrics: selectedMetrics,
    };

    if (detectionType === 'Model Bias Detection' && modelInputMode === 'train_model') {
      payload.model_type = modelType;
    }

    if (detectionType === 'Model Bias Detection' && modelInputMode === 'upload_pred_dataset') {
      if (!this.predictionData) {
        Utils.showError('Please upload a prediction dataset for Model Bias Detection.');
        return;
      }

      payload.dataset_pred = this.predictionData;

      const idCol = document.getElementById('id-col').value;
      const predLabelCol = document.getElementById('pred-label-col').value;
      const predProbaCol = document.getElementById('pred-proba-col').value;
      const probaThreshold = parseFloat(document.getElementById('proba-threshold').value);

      if (!predLabelCol) {
        Utils.showError('Please select the predicted label column.');
        return;
      }

      if (idCol) payload.id_column = idCol;
      payload.pred_label_col = predLabelCol;
      if (predProbaCol) payload.pred_proba_col = predProbaCol;
      payload.proba_threshold = probaThreshold;
    }

    console.log('Analysis payload:', payload);

    try {
      this.setLoading(true);
      const response = await API.analyze(payload);

      this.analysisResult = response.results;
      this.lastAnalyzeResponse = response;
      this.lastAnalysisConfig = {
        detection_type: detectionType,
        protected_attr: protectedAttr,
        label_column: labelCol,
        privileged_value: privVal,
        unprivileged_value: unprivVal,
        selected_metrics: selectedMetrics,
        model_type: modelType,
      };
      this.displayAnalysisResults(response);

      Utils.showSuccess('Analysis completed!');
    } catch (error) {
      Utils.showError(error);
    } finally {
      this.setLoading(false);
    }
  }

  displayAnalysisResults(response = null) {
    const container = document.getElementById('analysis-results');
    container.classList.remove('hidden');

    // Display raw results
    // Display raw results
    document.getElementById('results-json').textContent = JSON.stringify(this.analysisResult, null, 2);

    // Render Visualizations
    const detectionType = document.getElementById('detection-type').value;
    const metricsDef = detectionType === 'Dataset Bias Detection' ?
      (this.metrics.dataset_metrics || {}) :
      (this.metrics.classification_metrics || {});

    // Clear previous visualizations if any
    ['viz-group-distribution', 'viz-fairness-dashboard', 'viz-metric-comparison', 'viz-detailed-metrics'].forEach(id => {
      document.getElementById(id).innerHTML = '';
    });

    Visualization.renderAnalysisVisualizations(this.analysisResult, metricsDef);

    const modelComparisonSection = document.getElementById('model-comparison-section');
    if (detectionType === 'Model Bias Detection' && response && response.all_model_results) {
      modelComparisonSection.classList.remove('hidden');
      Visualization.renderModelComparison(
        'viz-model-comparison',
        response.all_model_results,
        this.metrics.classification_metrics || {},
        this.modelComparisonMetrics
      );
    } else {
      modelComparisonSection.classList.add('hidden');
      document.getElementById('viz-model-comparison').innerHTML = '';
    }

    // Display suggestion
    if (detectionType === 'Dataset Bias Detection') {
      const suggestion = Utils.computeSuggestion(this.analysisResult);
      if (suggestion) {
        const suggestionHtml = `
          <strong>Suggested Mitigation Level:</strong> ${suggestion.level}<br>
          <strong>Recommended Methods:</strong> ${suggestion.methods.join(', ')}<br>
          <strong>Reason:</strong> ${suggestion.reason}
        `;
        document.getElementById('suggestion-text').innerHTML = suggestionHtml;
        document.getElementById('suggestion-box').classList.remove('hidden');
      }
    }
  }

  async runMitigation() {
    if (!this.sessionId) {
      Utils.showError('Please upload a dataset first');
      return;
    }

    const detectionType = document.getElementById('detection-type').value;

    try {
      this.setLoading(true);
      if (detectionType === 'Model Bias Detection') {
        if (!this.lastAnalysisConfig) {
          Utils.showError('Run analysis once before model mitigation.');
          return;
        }

        const modelType = document.getElementById('mitigation-model-type').value;
        const payload = {
          session_id: this.sessionId,
          detection_type: 'Model Bias Detection',
          protected_attr: this.lastAnalysisConfig.protected_attr,
          label_column: this.lastAnalysisConfig.label_column,
          privileged_value: this.lastAnalysisConfig.privileged_value,
          unprivileged_value: this.lastAnalysisConfig.unprivileged_value,
          model_type: modelType,
          selected_metrics: this.modelComparisonMetrics,
        };
        

        console.log('Model mitigation payload:', payload);
        const response = await API.analyze(payload);

        document.getElementById('mitigation-results').classList.remove('hidden');
        document.getElementById('mitigation-json').textContent = JSON.stringify(response, null, 2);
        document.getElementById('download-section').classList.add('hidden');
        document.getElementById('viz-mitigation-comparison').innerHTML = '';

        Visualization.renderModelComparison(
          'viz-model-mitigation-comparison',
          response.all_model_results || {},
          this.metrics.classification_metrics || {},
          this.modelComparisonMetrics
        );

        Utils.showSuccess('Model mitigation comparison completed.');
      } else {
        const method = document.getElementById('mitigation-method').value;
        const payload = {
          session_id: this.sessionId,
          method: method,
          kwargs: {},
        };

        if (method === 'disparate_impact_remover') {
          payload.kwargs.repair_level = parseFloat(document.getElementById('repair-level').value);
        }

        console.log('Dataset mitigation payload:', payload);
        const response = await API.mitigate(payload);

        document.getElementById('mitigation-results').classList.remove('hidden');
        document.getElementById('mitigation-json').textContent = JSON.stringify(response.new_results, null, 2);
        document.getElementById('viz-model-mitigation-comparison').innerHTML = '';

        const metricsDef = this.metrics.dataset_metrics || {};
        if (this.analysisResult) {
          Visualization.renderMitigationComparison(this.analysisResult, response.new_results, metricsDef);
        }

        if (response.download_endpoint) {
          const downloadUrl = API.getDownloadUrl(response.download_endpoint);
          document.getElementById('download-link').href = downloadUrl;
          document.getElementById('download-section').classList.remove('hidden');
        }

        Utils.showSuccess('Dataset mitigation applied successfully.');
      }
    } catch (error) {
      Utils.showError(error);
    } finally {
      this.setLoading(false);
    }
  }

  clearResults() {
    document.getElementById('mitigation-results').classList.add('hidden');
    document.getElementById('download-section').classList.add('hidden');
    document.getElementById('viz-model-mitigation-comparison').innerHTML = '';
    Utils.showSuccess('Results cleared');
  }


  setLoading(isLoading) {
    const buttons = document.querySelectorAll('button');
    buttons.forEach(btn => {
      btn.disabled = isLoading;
    });
  }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new BiasDetectionApp();
});
