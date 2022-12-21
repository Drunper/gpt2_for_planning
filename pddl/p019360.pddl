(define (problem logistics-test)
(:domain logistics)
(:objects
	apn6 - airplane
	cit2 cit1 - city
	apt8 apt2 - airport
	tru3 tru5 - truck
	obj21 obj55 obj13 obj66 obj22 obj77 - package
	pos44 pos55 pos21 pos11 pos23 pos66 - location
)
(:init
	(at apn6 apt8)
	(at obj21 pos23)
	(at obj55 pos21)
	(at obj13 pos21)
	(at obj66 pos23)
	(at obj22 pos55)
	(at obj77 pos11)
	(in-city apt8 cit1)
	(in-city apt2 cit2)
	(in-city pos44 cit2)
	(in-city pos55 cit1)
	(in-city pos21 cit1)
	(in-city pos11 cit2)
	(in-city pos23 cit2)
	(in-city pos66 cit2)
	(at tru5 pos23)
	(at tru3 pos55)
)
(:goal
	(and
		(at obj77 pos44)
		(at obj66 pos66)
))
)
