export const Position = {

  TOPLEFT: 0,

  TOPCENTER: 1,

  TOPRIGHT: 2,

  MIDDLELEFT: 3,

  MIDDLECENTER: 4,

  MIDDLERIGHT: 5,

  BOTTOMLEFT: 6,

  BOTTOMCENTER: 7,

  BOTTOMRIGHT: 8,

} as const;



export type Position = (typeof Position)[keyof typeof Position];



/** [boardId, cell] — board and cell are each 0–8 */

export type Move = [number, number];



/** 0 = empty, 1 = X, -1 = O */

export type Cell = 0 | 1 | -1;



export type CellValue = Player | 0;



/** 1 = X, -1 = O */

export type Player = 1 | -1;



/** Flat 81-cell board state */

export type GameState = number[];



/** Flat 9-cell meta board: 0 ongoing, 1/-1 winner, 2 draw */

export type MetaState = number[];



/** 0 = ongoing, 1 | -1 = winner, 2 = draw */

export type SmallBoardResult = CellValue | 2;


