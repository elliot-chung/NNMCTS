import type { Move, UTTTGame } from "@/lib/uttt";

export interface MctsEnvironment<TMove = Move> {
  copy(): MctsEnvironment<TMove>;
  validMoves(): TMove[];
  makeMove(move: TMove): MctsEnvironment<TMove>;
  isTerminal(): boolean;
  getWinner(): number;
  currentTurn(): number;
  getState(): readonly number[];
  translate(move: TMove): number;
  getMask(): number[];
  getCanonicalState(): [number[], number[]];
}

export type UTTTEnvironment = UTTTGame & MctsEnvironment;

export interface ExplorePerf {
  traverseTime: number;
  rolloutTime: number;
  updateTime: number;
  createTime: number;
}

export interface MctsOptions {
  iters?: number;
  showExecutionTime?: boolean;
}

export interface MctsResult<TMove = Move> {
  move: TMove;
  policy: number[];
}
