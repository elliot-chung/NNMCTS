import { describe, expect, it } from "vitest";

import { UTTTGame } from "./game";

import { Position } from "./types";



function fillBoard(

  game: UTTTGame,

  boardId: number,

  cells: number[],

): void {

  for (let i = 0; i < 9; i++) {

    game.state[boardId * 9 + i] = cells[i];

  }

  game.metaState[boardId] = UTTTGame.getWinnerSmallBoard(

    game.state.slice(boardId * 9, boardId * 9 + 9),

  );

}



describe("UTTTGame", () => {

  it("allows first move anywhere on any unfinished board", () => {

    const game = new UTTTGame();

    const moves = game.validMoves();



    expect(moves).toHaveLength(81);

    expect(moves).toContainEqual([0, 0]);

    expect(moves).toContainEqual([4, 4]);

    expect(moves).toContainEqual([8, 8]);

  });



  it("forces the next move onto the board chosen by the previous cell", () => {

    const game = new UTTTGame().makeMove([4, Position.MIDDLECENTER]);



    const moves = game.validMoves();

    expect(moves.every((m) => m[0] === Position.MIDDLECENTER)).toBe(true);

    expect(moves).toHaveLength(8);

    expect(moves).not.toContainEqual([

      Position.MIDDLECENTER,

      Position.MIDDLECENTER,

    ]);

  });



  it("allows free choice when the target board is already finished", () => {

    const game = new UTTTGame();

    fillBoard(game, Position.TOPLEFT, [1, 1, 1, -1, -1, 0, 0, 0, 0]);

    game.previousMove = [Position.MIDDLECENTER, Position.TOPLEFT];

    game.turn = 1;



    const moves = game.validMoves();

    const targetBoards = new Set(moves.map((m) => m[0]));



    expect(targetBoards.has(Position.TOPLEFT)).toBe(false);

    expect(moves.length).toBeGreaterThan(8);

    expect(moves.some((m) => m[0] !== Position.TOPLEFT)).toBe(true);

    for (const move of moves) {
      expect(game.isValid(move)).toBe(true);
    }

  });



  it("falls back to free moves when the forced board has no open cells", () => {

    const game = new UTTTGame();

    fillBoard(

      game,

      Position.TOPLEFT,

      [1, -1, 1, -1, 1, -1, -1, 1, -1],

    );

    game.metaState[Position.TOPLEFT] = 0;

    game.previousMove = [Position.MIDDLECENTER, Position.TOPLEFT];

    game.turn = 1;



    const moves = game.validMoves();



    expect(moves.length).toBeGreaterThan(0);

    expect(moves.every((move) => move[0] !== Position.TOPLEFT)).toBe(true);

    expect(() => game.isTerminal()).not.toThrow();

    for (const move of moves) {
      expect(game.isValid(move)).toBe(true);
    }

  });



  it("detects mini-board wins and draws", () => {

    expect(

      UTTTGame.getWinnerSmallBoard([1, 1, 1, 0, -1, 0, 0, 0, 0]),

    ).toBe(1);

    expect(

      UTTTGame.getWinnerSmallBoard([-1, 0, 0, -1, 0, 0, -1, 0, 0]),

    ).toBe(-1);

    expect(

      UTTTGame.getWinnerSmallBoard([1, -1, 1, -1, 1, -1, -1, 1, -1]),

    ).toBe(2);

    expect(

      UTTTGame.getWinnerSmallBoard([0, 0, 0, 0, 0, 0, 0, 0, 0]),

    ).toBe(0);

  });



  it("detects meta-board wins", () => {

    const game = new UTTTGame();

    for (let boardId = 0; boardId < 3; boardId++) {

      fillBoard(game, boardId, [1, 1, 1, 0, 0, 0, 0, 0, 0]);

    }

    game.metaState = [1, 1, 1, 0, 0, 0, 0, 0, 0];



    expect(game.getWinner()).toBe(1);

    expect(game.isTerminal()).toBe(true);

  });



  it("returns canonical state and mask aligned with valid moves", () => {

    const game = new UTTTGame().makeMove([2, Position.BOTTOMRIGHT]);



    const [normState, mask] = game.getCanonicalState();

    const validMoves = game.validMoves();



    expect(normState).toHaveLength(81);

    expect(mask).toHaveLength(81);

    expect(mask.filter((v) => v === 1)).toHaveLength(validMoves.length);



    for (const move of validMoves) {

      expect(mask[game.translate(move)]).toBe(1);

    }



    expect(normState[game.translate([2, Position.BOTTOMRIGHT])]).toBe(-1);

  });



  it("flips canonical perspective for the second player", () => {

    const game = new UTTTGame().makeMove([0, 0]).makeMove([0, 1]);



    const [normState] = game.getCanonicalState();

    expect(normState[0]).toBe(1);

    expect(normState[1]).toBe(-1);

  });



  it("copy produces an independent game instance", () => {

    const game = new UTTTGame().makeMove([3, 3]);

    const copy = game.copy();

    const movedCopy = copy.makeMove([3, 4]);



    expect(game.getState()[game.translate([3, 4])]).toBe(0);

    expect(movedCopy.getState()[movedCopy.translate([3, 4])]).toBe(-1);

    expect(movedCopy.currentTurn()).toBe(1);

    expect(game.currentTurn()).toBe(-1);

  });



  it("treats meta draw as no winner", () => {

    const metaState = [2, 2, 2, 2, 2, 2, 2, 2, 2];

    const game = new UTTTGame(Array(81).fill(0), 1, null, metaState);



    expect(game.getWinner()).toBe(0);

  });

});


