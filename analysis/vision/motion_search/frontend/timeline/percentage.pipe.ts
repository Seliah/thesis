import { Pipe, PipeTransform } from '@angular/core';

/** Pipe that allows values to be converted into percentage values. */
@Pipe({
    name: 'percentage'
})
export class PercentagePipe implements PipeTransform {
    transform(relative: Array<any>, type: string, scopeStart: number, scopeEnd: number, cssReady = false, revert = false): any {
        switch (type) {
            case 'width':
                if (relative.length != 2) {
                    throw new Error(`The given tuple for '${type}' does not have the required length of 1.`);
                }
                const width = (relative[1] - relative[0]) / (scopeEnd - scopeStart);
                return cssReady ? `${width * 100}%` : width;
            case 'position':
                if (relative.length != 1) {
                    throw new Error(`The given tuple for '${type}' does not have the required length of 1.`);
                }
                const leftPart = (relative[0] - scopeStart) / (scopeEnd - scopeStart);
                const part = revert ? leftPart : 1 - leftPart;
                return cssReady ? `${part * 100}%` : part;

            // Ensure type is correctly specified
            default:
                throw new Error(`'${type} is not a valid typw for PercentagePipe'`);
        }
    }
}
