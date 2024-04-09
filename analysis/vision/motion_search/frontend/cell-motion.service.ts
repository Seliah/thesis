import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Subject, concat, firstValueFrom, map, of } from 'rxjs';
import { environment } from 'share';
import { Selection } from './selection-overlay/selection-overlay.component';

export type SearchFrame = [start: number, end: number]


/**
 * Function to find consecutive ranges in an integer list.
 * @see https://stackoverflow.com/questions/2270910/how-to-reduce-consecutive-integers-in-an-array-to-hyphenated-range-expressions
 */
const consecutiveRanges = (integers: number[]) => {
  var ranges: SearchFrame[] = [], rstart, rend;
  for (var i = 0; i < integers.length; i++) {
    rstart = integers[i];
    rend = rstart;
    while (integers[i + 1] - integers[i] == 1) {
      // increment the index if the numbers sequential
      rend = integers[i + 1];
      i++;
    }
    ranges.push(rstart == rend ? [rstart, rstart] : [rstart, rend]);
  }
  return ranges;
}


const URL_HEATMAP = `${environment.services.analysis}/heatmap`;
const URL_MOTIONS = `${environment.services.analysis}/motion_data`;
/**
 * Size of the individual time spans to show.
 *
 * This cannot be lower than 1. Using 1 will result in CSS problems as the dicts get to small.
 */
export const SPAN_SIZE = 30;

/** Service that allows to communicate with the motion api to get motion information about a given camera. */
@Injectable({
    providedIn: 'root'
})
export class CellMotionService {
    private readonly searchFrames: Record<string, SearchFrame[]> = {};

    private readonly selection$ = new Subject<void>();

    constructor(private readonly http: HttpClient) {}

    async onDragSelect(cameraId: string, selection: Selection) {
        const params = {
            camera_id: cameraId,
            height: round(selection.height, 4),
            left: round(selection.position.x, 4),
            span_size: SPAN_SIZE,
            top: round(selection.position.y, 4),
            width: round(selection.width, 4)
        };
        const request = this.http.get<number[]>(URL_MOTIONS, { params });
        const response = await firstValueFrom(request);
        this.searchFrames[cameraId] = consecutiveRanges(response);
        this.selection$.next();
    }

    unselect = (cameraId: string) => {
        this.searchFrames[cameraId] = [];
        this.selection$.next();
    };

    getSearchFrames$(cameraId: string) {
        const current$ = of(this.searchFrames[cameraId]);
        const next$ = this.selection$.pipe(map(() => this.searchFrames[cameraId] ?? []));
        return concat(current$, next$);
    }

    async getHeatmap$(cameraId: string) {
        const params = { camera_id: cameraId };
        const request = this.http.get<number[]>(URL_HEATMAP, { params });
        return firstValueFrom(request);
    }
}
