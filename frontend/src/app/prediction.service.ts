import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface FeatureSpec {
  name: string;
  label: string;
  group: string;
  description: string;
  kind: 'number' | 'rate' | 'binary' | 'category';
  integer?: boolean;
  min_allowed?: number;
  max_allowed?: number;
  median?: number;
  options?: string[];
}

export interface Schema {
  schema_version: number;
  model: string;
  protocol: string;
  threshold: number;
  n_features: number;
  groups: string[];
  features: FeatureSpec[];
}

export interface PredictionResponse {
  prediction: 'normal' | 'attack';
  label: number;
  probabilities: [number, number];
  threshold: number;
  warnings: string[];
  model?: string;
  disclaimer?: string;
}

@Injectable({ providedIn: 'root' })
export class PredictionService {
  private readonly http = inject(HttpClient);

  loadSchema(): Observable<Schema> {
    return new Observable(subscriber => {
      fetch('/assets/schema.json')
        .then(response => {
          if (!response.ok) {
            throw new Error(`Schema request failed (${response.status})`);
          }
          return response.json() as Promise<Schema>;
        })
        .then(schema => {
          if (!Array.isArray(schema.features) || !Array.isArray(schema.groups)) {
            throw new Error('Schema response is missing features or groups.');
          }
          subscriber.next(schema);
          subscriber.complete();
        })
        .catch(error => subscriber.error(error));
    });
  }

  loadExamples(): Observable<Record<string, Record<string, string | number>>> {
    return new Observable(subscriber => {
      Promise.all([
        fetch('/assets/test_payload_normal.json').then(response => response.json()),
        fetch('/assets/test_payload.json').then(response => response.json()),
      ]).then(([normal, attack]) => {
        subscriber.next({ normal, attack });
        subscriber.complete();
      }).catch(error => subscriber.error(error));
    });
  }

  predict(payload: Record<string, string | number>): Observable<PredictionResponse> {
    return this.http.post<PredictionResponse>('/api/predict', payload);
  }
}
