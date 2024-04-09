import { CommonModule } from '@angular/common';
import { ChangeDetectionStrategy, Component, Input, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { BehaviorSubject, Subject, map } from 'rxjs';
import { CellMotionService } from '../cell-motion.service';

/** Component that creates a transparent overlay with heatmap information for the given camera. */
@Component({
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [CommonModule, MatCheckboxModule, FormsModule],
    selector: 'app-heatmap',
    standalone: true,
    styles: [
        `
            :host {
                display: block;
                pointer-events: none;
            }

            #wrapper {
                grid-template-columns: repeat(16, auto);
            }
        `
    ],
    template: `
        <div class="absolute left-2 bottom-2 pointer-events-auto"><mat-checkbox (change)="show$.next($event.checked)">Show Heatmap</mat-checkbox></div>
        <div id="wrapper" *ngIf="(show$ | async) && (heatmap$ | async)?.length" class="size-full grid grid-cols-16">
            <div *ngFor="let n of heatmap$ | async" [style.opacity]="((1 / CAP) * n) / 1.5" class="bg-accent"></div>
        </div>
    `
})
export class HeatmapComponent implements OnInit {
    readonly CAP = 1000;

    @Input() cameraId: string;

    max$ = new Subject<number>();

    show$ = new BehaviorSubject(false);

    heatmap$ = new BehaviorSubject<number[]>([]);

    constructor(private readonly cellMotion: CellMotionService) {}

    async ngOnInit() {
        this.heatmap$.pipe(map((heatmap) => Math.max(...heatmap))).subscribe(this.max$);
        const heatmap = await this.cellMotion.getHeatmap$(this.cameraId);
        this.heatmap$.next(heatmap);
    }
}
