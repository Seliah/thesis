import { CommonModule } from '@angular/common';
import { NgModule } from '@angular/core';
import { TimelineComponent } from './timeline.component';
import { PercentagePipe } from './percentage.pipe';

@NgModule({
    declarations: [TimelineComponent],
    exports: [TimelineComponent],
    imports: [CommonModule, PercentagePipe]
})
export class TimelineModule {}
