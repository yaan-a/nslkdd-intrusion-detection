import { CommonModule } from '@angular/common';
import { ChangeDetectorRef, Component, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { PredictionService, FeatureSpec, PredictionResponse, Schema } from './prediction.service';

@Component({
  imports: [CommonModule, FormsModule],
  selector: 'app-root',
  styleUrl: './app.css',
  templateUrl: './app.html',
})
export class App {
  private readonly predictionService = inject(PredictionService);
  private readonly changeDetector = inject(ChangeDetectorRef);

  schema: Schema | null = null;
  values: Record<string, string | number | null> = {};
  examples: Record<string, Record<string, string | number>> = {};
  serviceSearch = '';
  loading = true;
  submitting = false;
  error = '';
  result: PredictionResponse | null = null;
  readonly deployedStatic = !['localhost', '127.0.0.1'].includes(window.location.hostname);

  constructor() {
    this.predictionService.loadSchema().subscribe({
      next: schema => {
        try {
          this.schema = schema;
          this.initializeValues(schema.features);
          this.loading = false;
        } catch (error) {
          this.error = error instanceof Error ? error.message : 'Invalid model schema.';
          this.loading = false;
        }
        this.changeDetector.detectChanges();
      },
      error: () => {
        this.error = 'Could not load the model input schema.';
        this.loading = false;
        this.changeDetector.detectChanges();
      },
    });
    this.predictionService.loadExamples().subscribe({
      next: examples => {
        this.examples = examples;
        this.changeDetector.detectChanges();
      },
    });
  }

  initializeValues(features: FeatureSpec[]) {
    for (const feature of features) {
      this.values[feature.name] = feature.kind === 'category'
        ? feature.options?.[0] ?? ''
        : feature.median ?? 0;
    }
  }

  groupedFeatures(group: string) {
    return this.schema?.features.filter(feature => feature.group === group) ?? [];
  }

  serviceOptions(feature: FeatureSpec) {
    if (feature.name !== 'service' || !feature.options) {
      return feature.options ?? [];
    }
    const query = this.serviceSearch.trim().toLowerCase();
    return query
      ? feature.options.filter(option => option.toLowerCase().includes(query))
      : feature.options;
  }

  loadExample(name: string) {
    const example = this.examples[name];
    if (!example) return;
    this.values = { ...example };
    this.result = null;
    this.error = '';
  }

  submit() {
    if (!this.schema || this.submitting) return;
    if (this.deployedStatic) {
      this.error = 'Static GitHub Pages demo: start Flask and Angular locally to run predictions.';
      return;
    }
    this.error = '';
    this.result = null;
    const payload: Record<string, string | number> = {};
    for (const feature of this.schema.features) {
      const value = this.values[feature.name];
      if (value === null || value === '') {
        this.error = `Enter a value for ${feature.label}.`;
        return;
      }
      payload[feature.name] = feature.kind === 'category'
        ? String(value)
        : Number(value);
    }
    this.submitting = true;
    this.predictionService.predict(payload).subscribe({
      next: response => {
        this.result = response;
        this.submitting = false;
        this.changeDetector.detectChanges();
      },
      error: response => {
        this.error = response.error?.problems?.join(' ') ||
          response.error?.error || 'The API could not process this record.';
        this.submitting = false;
        this.changeDetector.detectChanges();
      },
    });
  }

  trackFeature(_index: number, feature: FeatureSpec) {
    return feature.name;
  }
}
