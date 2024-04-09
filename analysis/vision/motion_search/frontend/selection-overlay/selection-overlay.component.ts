import { CdkDrag, CdkDragHandle, DragDropModule, Point } from '@angular/cdk/drag-drop';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { ChangeDetectionStrategy, Component, ElementRef, EventEmitter, HostListener, Output, ViewChild } from '@angular/core';
import { ResizableModule, ResizeEvent, ResizeHandleDirective } from 'angular-resizable-element';
import { BehaviorSubject } from 'rxjs';

export type Selection = { position: Point; width: number; height: number };

/** Component that makes it possible to select a rectangle inside the host component with point and drag. */
@Component({
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [CommonModule, CdkDrag, CdkDragHandle, ResizableModule, HttpClientModule, DragDropModule],
    selector: 'app-selection-overlay',
    standalone: true,
    styleUrls: ['./selection-overlay.component.scss'],
    template: `
        <!-- Background -->
        <svg width="100%" height="100%" [style.visibility]="(selectionActive$ | async) ? 'inherit' : 'hidden'">
            <mask id="test">
                <g id="text" fill="white">
                    <rect width="100%" height="100%" fill="#999"></rect>
                    <rect [style]="maskStyle$ | async" rx="15" ry="15" fill="black" class="!visible"></rect>
                </g>
            </mask>
            <rect width="100%" height="100%" mask="url(#test)" fill="black"></rect>
        </svg>
        <!-- Selection element -->
        <div
            #selection
            [ngStyle]="selectionStyle$ | async"
            class="selectionBox"
            [class.active]="(selectionActive$ | async) === false"
            cdkDragBoundary=".info"
            cdkDrag
            #drag="cdkDrag"
            mwlResizable
            [cdkDragFreeDragPosition]="position$ | async"
            (cdkDragMoved)="maskStyle$.next(selection.style.cssText)"
            (cdkDragEnded)="maskStyle$.next(selection.style.cssText); onMove(selection, drag)"
            (resizing)="onResize($event, drag)"
            (resizeEnd)="onResizeEnd($event, drag)"
            [enableGhostResize]="true"
            [validateResize]="validate(drag)"
        >
            <div class="cursor-move size-full" cdkDragHandle></div>
            <div
                class="absolute size-8 -bottom-4 -right-4 rounded-full shadow-lg cursor-se-resize resizeHandle"
                [class.active]="(selectionActive$ | async) === false"
                mwlResizeHandle
                [resizeEdges]="{ bottom: true, right: true }"
            ></div>
        </div>
        <div *ngIf="(selectionActive$ | async) === false" class="absolute size-full cursor-crosshair" (mousedown)="onMouseDown($event)"></div>
    `
})
export class SelectionOverlayComponent {
    private readonly DEFAULT_SIZE = {
        height: '0px',
        width: '0px'
    };

    @Output() readonly selected = new EventEmitter<Selection>();

    @Output() readonly unselected = new EventEmitter<void>();

    readonly maskStyle$ = new BehaviorSubject('');

    readonly selectionActive$ = new BehaviorSubject(false);

    readonly position$ = new BehaviorSubject<Point>({ x: 0, y: 0 });

    readonly selectionStyle$ = new BehaviorSubject<{ width: string; height: string }>(this.DEFAULT_SIZE);

    @ViewChild(ResizeHandleDirective) private resizeHandle!: ResizeHandleDirective;

    constructor(private readonly hostElement: ElementRef<HTMLElement>) {}

    onMove(selection: HTMLDivElement, drag: CdkDrag) {
        console.debug('Moved selection.');
        const position = drag.getFreeDragPosition();
        this.emit(selection.offsetHeight, selection.offsetWidth, position);
    }

    private emit(height: number, width: number, position: Point) {
        console.debug(`Emitting new selection: position: (${position.x}, ${position.y}); width: ${width}; height: ${height}`);
        const element = this.hostElement.nativeElement;
        this.selected.emit({
            height: height / element.offsetHeight,
            position: {
                x: position.x / element.offsetWidth,
                y: position.y / element.offsetHeight
            },
            width: width / element.offsetWidth
        });
    }

    validate(drag: CdkDrag) {
        const element = this.hostElement.nativeElement;
        const position = drag.getFreeDragPosition();
        return (event: ResizeEvent) => position.x + event.rectangle.width < element.offsetWidth && position.y + event.rectangle.height < element.offsetHeight;
    }

    onResizeEnd(event: ResizeEvent, drag: CdkDrag) {
        console.debug('Resized selection.');
        const position = drag.getFreeDragPosition();
        this.selectionStyle$.next({
            height: `${event.rectangle.height}px`,
            width: `${event.rectangle.width}px`
        });
        this.emit(event.rectangle.height, event.rectangle.width, position);
    }

    onResize(event: ResizeEvent, drag: CdkDrag) {
        const position = drag.getFreeDragPosition();
        const style = `height: ${event.rectangle.height}px; width: ${event.rectangle.width}px; transform: translate3d(${position.x}px, ${position.y}px, 0px); visibility: inherit;`;
        this.maskStyle$.next(style);
    }

    onMouseDown(event: MouseEvent) {
        console.debug('Mouse down, starting rectangle selection');
        this.selectionActive$.next(true);
        this.position$.next({ x: event.offsetX, y: event.offsetY });
        this.resizeHandle.onMousedown(event, event.clientX, event.clientY);
    }

    @HostListener('contextmenu', ['$event']) onRightClick(event: MouseEvent) {
        console.debug('Right click, cancelling selection.');
        event.preventDefault();
        this.selectionActive$.next(false);
        this.selectionStyle$.next(this.DEFAULT_SIZE);
        this.unselected.emit();
    }
}
