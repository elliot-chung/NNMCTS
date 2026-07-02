import type {

  CellValue,

  GameState,

  MetaState,

  Move,

  Player,

  SmallBoardResult,

} from "./types";

import { Position } from "./types";



export class UTTTGame {

  readonly state: number[];

  readonly metaState: number[];

  readonly turn: Player;

  readonly previousMove: Move | null;



  constructor(

    state?: GameState | null,

    player: Player = 1,

    previousMove: Move | null = null,

    metaState?: MetaState | null,

  ) {

    if (player !== 1 && player !== -1) {

      throw new Error("Invalid player");

    }



    this.turn = player;

    this.previousMove = previousMove;



    if (state === null || state === undefined) {

      this.state = Array(81).fill(0);

      this.metaState = Array(9).fill(0);

    } else {

      if (state.length !== 81) {

        throw new Error("Invalid state");

      }

      this.state = [...state];

      this.metaState =

        metaState === null || metaState === undefined

          ? UTTTGame.calculateMetaState(this.state)

          : [...metaState];

    }

  }



  static translate(move: Move): number {

    const [boardId, cellId] = move;

    return boardId * 9 + cellId;

  }



  static getWinnerSmallBoard(cells: readonly number[]): SmallBoardResult {

    let occupant = cells[Position.TOPLEFT];

    if (occupant !== 0 && occupant !== 2) {

      if (

        cells[Position.TOPCENTER] === occupant &&

        cells[Position.TOPRIGHT] === occupant

      ) {

        return occupant as Player;

      }

      if (

        cells[Position.MIDDLELEFT] === occupant &&

        cells[Position.BOTTOMLEFT] === occupant

      ) {

        return occupant as Player;

      }

    }



    occupant = cells[Position.BOTTOMRIGHT];

    if (occupant !== 0 && occupant !== 2) {

      if (

        cells[Position.BOTTOMCENTER] === occupant &&

        cells[Position.BOTTOMLEFT] === occupant

      ) {

        return occupant as Player;

      }

      if (

        cells[Position.MIDDLERIGHT] === occupant &&

        cells[Position.TOPRIGHT] === occupant

      ) {

        return occupant as Player;

      }

    }



    occupant = cells[Position.MIDDLECENTER];

    if (occupant !== 0 && occupant !== 2) {

      if (

        cells[Position.MIDDLELEFT] === occupant &&

        cells[Position.MIDDLERIGHT] === occupant

      ) {

        return occupant as Player;

      }

      if (

        cells[Position.TOPLEFT] === occupant &&

        cells[Position.BOTTOMRIGHT] === occupant

      ) {

        return occupant as Player;

      }

      if (

        cells[Position.TOPCENTER] === occupant &&

        cells[Position.BOTTOMCENTER] === occupant

      ) {

        return occupant as Player;

      }

      if (

        cells[Position.TOPRIGHT] === occupant &&

        cells[Position.BOTTOMLEFT] === occupant

      ) {

        return occupant as Player;

      }

    }



    const emptySlots = cells.filter((value) => value === 0);

    if (emptySlots.length === 0) {

      return 2;

    }



    return 0;

  }



  static calculateMetaState(state: readonly number[]): MetaState {

    const metaState: MetaState = Array(9).fill(0);

    for (let i = 0; i < 9; i++) {

      metaState[i] = UTTTGame.getWinnerSmallBoard(state.slice(i * 9, i * 9 + 9));

    }

    return metaState;

  }



  currentTurn(): Player {

    return this.turn;

  }



  getState(): readonly number[] {

    return this.state;

  }



  translate(move: Move): number {

    return UTTTGame.translate(move);

  }



  getMask(): number[] {

    const posMask = Array(this.state.length).fill(0);

    for (const move of this.validMoves()) {

      posMask[this.translate(move)] = 1;

    }

    return posMask;

  }



  getCanonicalState(): [number[], number[]] {

    const normState = this.state.map((cell) => cell * this.turn);

    return [normState, this.getMask()];

  }



  isValid(move: Move): boolean {

    const [boardId, cellId] = move;

    if (boardId < 0 || boardId > 8 || cellId < 0 || cellId > 8) {

      return false;

    }



    const idx = UTTTGame.translate(move);

    if (this.previousMove === null) {

      return this.state[idx] === 0;

    }

    if (move[0] === this.previousMove[1]) {

      return this.state[idx] === 0;

    }

    return false;

  }



  validMoves(): Move[] {

    const finishedBoards = this.metaState

      .map((value, index) => (value !== 0 ? index : -1))

      .filter((index) => index !== -1);



    const forcedBoard =

      this.previousMove !== null ? this.previousMove[1] : null;



    if (forcedBoard === null || this.metaState[forcedBoard] !== 0) {

      const moves: Move[] = [];

      for (let boardId = 0; boardId < 9; boardId++) {

        if (finishedBoards.includes(boardId)) {

          continue;

        }

        for (let cellId = 0; cellId < 9; cellId++) {

          const idx = UTTTGame.translate([boardId, cellId]);

          if (this.state[idx] === 0) {

            moves.push([boardId, cellId]);

          }

        }

      }

      return moves;

    }



    const moves: Move[] = [];

    for (let cellId = 0; cellId < 9; cellId++) {

      const idx = UTTTGame.translate([forcedBoard, cellId]);

      if (this.state[idx] === 0) {

        moves.push([forcedBoard, cellId]);

      }

    }



    if (moves.length === 0) {

      throw new Error("No valid moves");

    }



    return moves;

  }



  getWinner(): CellValue {

    const result = UTTTGame.getWinnerSmallBoard(this.metaState);

    return result === 2 ? 0 : (result as CellValue);

  }



  isTerminal(): boolean {

    return this.getWinner() !== 0 || this.validMoves().length === 0;

  }



  makeMove(move: Move): UTTTGame {

    const nextState = [...this.state];

    nextState[UTTTGame.translate(move)] = this.turn;



    const nextMetaState = [...this.metaState];

    const boardId = move[0];

    if (nextMetaState[boardId] !== 0) {

      throw new Error("Board already won");

    }

    nextMetaState[boardId] = UTTTGame.getWinnerSmallBoard(

      nextState.slice(boardId * 9, boardId * 9 + 9),

    );



    return new UTTTGame(

      nextState,

      (this.turn * -1) as Player,

      move,

      nextMetaState,

    );

  }



  makeRandomMove(): UTTTGame {

    const moves = this.validMoves();

    const move = moves[Math.floor(Math.random() * moves.length)];

    return this.makeMove(move);

  }



  copy(): UTTTGame {

    return new UTTTGame(

      [...this.state],

      this.turn,

      this.previousMove ? ([...this.previousMove] as Move) : null,

      [...this.metaState],

    );

  }

}


