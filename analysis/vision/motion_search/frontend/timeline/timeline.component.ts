import { ChangeDetectionStrategy, ChangeDetectorRef, Component, Input, OnInit } from '@angular/core';
import { BehaviorSubject, map } from 'rxjs';
import { CellMotionService, SPAN_SIZE } from '../cell-motion.service';

type Frame = {start: number, end: number}

function getTotalMinutesOfDay(date: Date) {
  return date.getMinutes() + date.getHours() * 60;
}

function getTotalSecondsOfDay(date: Date) {
  return date.getSeconds() + getTotalMinutesOfDay(date) * 60;
}

function getTotalMillisecondsSecondsOfDay(date: Date) {
  return date.getMilliseconds() + getTotalSecondsOfDay(date) * 1000;
}

/** Component that will display given frames as bars in a horizontal timeline. */
@Component({
    changeDetection: ChangeDetectionStrategy.OnPush,
    selector: 'app-timeline',
    styleUrls: ['./timeline.component.scss'],
    templateUrl: './timeline.component.html'
})
export class TimelineComponent implements OnInit {
    /** TODO rename to "id" */
    @Input() name: string;

    @Input() showName: boolean;

    @Input() showLine = true;

    @Input() start: number;

    @Input() end: number;

    @Input() frames: Frame[];

    /**
     * The name of the camera that is displayed in a label.
     */
    cameraName: string;

    searchFrames$ = new BehaviorSubject<SearchFrame[]>([]);

    readonly MOTION_SEARCH_START = 0;
    readonly MOTION_SEARCH_END = this.videoplayer.ceil.pipe(
        // Take millisecond to second conversion and span size into account
        map((ms) => getTotalMillisecondsSecondsOfDay(new Date(ms)) / (1000 * SPAN_SIZE))
    );

    constructor(
        readonly changeDetector: ChangeDetectorRef,
        private cameraService: CamerasService,
        private readonly cellMotion: CellMotionService,
        private readonly videoplayer: VideoplayerService,
    ) {}

    ngOnInit() {
        this.cameraName = this.cameraService.getCameraNameById(this.name);
        this.cellMotion.getSearchFrames$(this.name).subscribe(this.searchFrames$);
    }
}
