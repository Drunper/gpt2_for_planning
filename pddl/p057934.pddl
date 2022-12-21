(define (problem logistics-test)
(:domain logistics)
(:objects
	apn2 apn4 apn5 apn8 - airplane
	cit6 cit1 cit5 cit3 cit2 - city
	apt3 apt8 apt4 apt6 - airport
	tru1 tru2 tru5 tru3 tru4 - truck
	obj77 obj13 obj88 obj44 obj21 obj22 obj12 obj66 obj99 obj55 - package
	pos21 pos23 pos22 pos77 pos11 pos66 pos44 pos12 pos55 pos13 - location
)
(:init
	(at apn2 apt3)
	(at apn4 apt4)
	(at apn5 apt4)
	(at apn8 apt4)
	(at obj77 pos11)
	(at obj13 pos22)
	(at obj88 pos22)
	(at obj44 pos21)
	(at obj21 pos77)
	(at obj22 pos11)
	(at obj12 pos11)
	(at obj66 pos11)
	(at obj99 pos22)
	(at obj55 pos11)
	(in-city apt3 cit1)
	(in-city apt8 cit3)
	(in-city apt4 cit2)
	(in-city apt6 cit6)
	(in-city pos21 cit6)
	(in-city pos23 cit3)
	(in-city pos22 cit2)
	(in-city pos77 cit2)
	(in-city pos11 cit5)
	(in-city pos66 cit5)
	(in-city pos44 cit3)
	(in-city pos12 cit6)
	(in-city pos55 cit1)
	(in-city pos13 cit1)
	(at tru4 pos21)
	(at tru3 pos23)
	(at tru5 pos77)
	(at tru2 pos11)
	(at tru1 pos13)
)
(:goal
	(and
		(at obj21 pos13)
		(at obj77 pos66)
))
)
